// Multiplayer transport.
//
// Split transports (checklist: networking):
// * TCP  — reliable, ordered: handshake, voxel edits, join/leave, the bulk
//   edit-log replay, heartbeats. Edits MUST NOT be dropped or reordered.
// * UDP  — pose updates (PlayerUpdate). Lossy is fine: a dropped pose is
//   superseded by the next one, and head-of-line blocking on a laggy edit must
//   never stall movement.
//
// World sync is still via shared seed (every client regenerates identical
// terrain) + an authoritative, ordered, versioned edit log on the server.
//
// Threading: each TCP connection gets a reader + writer thread bridged to the
// main game/server thread via crossbeam channels; one UDP reader thread per
// endpoint. Mutable game state stays single-threaded on the main thread — no
// shared mutable locks except a small token map the server's UDP thread reads.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use serde::{Deserialize, Serialize};

pub type PlayerId = u32;

/// Wire protocol version. Bumped on any incompatible change to `Message`; the
/// server rejects clients whose version differs so a format change can never
/// silently corrupt state (checklist: wire protocol versioning).
pub const PROTOCOL_VERSION: u32 = 1;

/// Drop a client we haven't heard from (any transport) for this long.
pub const CLIENT_TIMEOUT: Duration = Duration::from_secs(8);
/// Client heartbeat interval (must be < CLIENT_TIMEOUT with margin).
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_millis(2000);

/// Render remote players this far in the past so the interpolation buffer
/// always has two samples to blend between (checklist: interpolation buffer).
pub const INTERP_DELAY: Duration = Duration::from_millis(100);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    /// Client → server on connect. Carries the protocol version.
    Hello { version: u32 },
    /// Server → client: rejected because of a version mismatch.
    VersionMismatch { server_version: u32 },
    /// Server → new client: player id, world seed, a UDP auth token, and the
    /// server's UDP port.
    JoinAck { your_id: PlayerId, seed: u64, token: u64, udp_port: u16 },
    /// Pose. Sent over UDP in steady state (also valid over TCP as a fallback).
    PlayerUpdate { id: PlayerId, pos: [f32; 3], yaw: f32, pitch: f32 },
    PlayerJoin { id: PlayerId },
    PlayerLeave { id: PlayerId },
    /// A single voxel edit, sequence-numbered for ordering/versioning.
    VoxelEdit { wx: i32, wy: i32, wz: i32, mat: u8, seq: u64 },
    /// Sphere-of-impact, sequence-numbered. One message instead of O(r³) edits.
    Explode { cx: i32, cy: i32, cz: i32, radius: u8, mat: u8, seq: u64 },
    /// Compressed bulk edit-log replay sent to a joiner (deflate of a
    /// bincode Vec<Message> of ordered edits). Fixes late-joiner desync.
    EditLog { compressed: Vec<u8> },
    /// Keep-alive.
    Heartbeat,
}

// ---------------- framing ----------------

fn read_message<R: Read>(r: &mut R) -> std::io::Result<Message> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "msg too big"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    bincode::deserialize(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn write_message<W: Write>(w: &mut W, msg: &Message) -> std::io::Result<()> {
    let buf = bincode::serialize(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    w.write_all(&(buf.len() as u32).to_le_bytes())?;
    w.write_all(&buf)?;
    w.flush()
}

// ---------------- compression (bulk edit-log replay) ----------------

/// Deflate-compress an ordered list of edit messages for the join replay.
pub fn compress_edits(edits: &[Message]) -> Vec<u8> {
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    let raw = bincode::serialize(edits).unwrap_or_default();
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::fast());
    let _ = enc.write_all(&raw);
    enc.finish().unwrap_or_default()
}

/// Inverse of `compress_edits`.
pub fn decompress_edits(compressed: &[u8]) -> Vec<Message> {
    use flate2::read::DeflateDecoder;
    let mut dec = DeflateDecoder::new(compressed);
    let mut raw = Vec::new();
    if dec.read_to_end(&mut raw).is_err() {
        return Vec::new();
    }
    bincode::deserialize(&raw).unwrap_or_default()
}

// ---------------- remote-player interpolation ----------------

#[derive(Clone, Copy)]
struct PoseSample {
    t: Instant,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

/// Per-remote-player interpolation buffer. Holds recent pose samples and
/// returns the pose interpolated `INTERP_DELAY` in the past, so other players
/// move smoothly instead of snapping to each received packet.
pub struct RemotePlayer {
    samples: std::collections::VecDeque<PoseSample>,
}

impl RemotePlayer {
    pub fn new() -> Self {
        Self { samples: std::collections::VecDeque::with_capacity(16) }
    }

    pub fn push_sample(&mut self, t: Instant, pos: [f32; 3], yaw: f32, pitch: f32) {
        // Drop out-of-order/duplicate stamps (UDP can reorder).
        if let Some(last) = self.samples.back() {
            if t <= last.t {
                return;
            }
        }
        self.samples.push_back(PoseSample { t, pos, yaw, pitch });
        while self.samples.len() > 16 {
            self.samples.pop_front();
        }
    }

    /// Interpolated pose at `now - INTERP_DELAY`. Returns (pos, yaw, pitch).
    pub fn sample(&self, now: Instant) -> Option<([f32; 3], f32, f32)> {
        if self.samples.is_empty() {
            return None;
        }
        let target = now.checked_sub(INTERP_DELAY).unwrap_or(now);
        // Find the pair (a, b) with a.t <= target <= b.t.
        let mut prev: Option<&PoseSample> = None;
        for s in &self.samples {
            if s.t >= target {
                if let Some(a) = prev {
                    let span = s.t.duration_since(a.t).as_secs_f32().max(1e-4);
                    let f = (target.duration_since(a.t).as_secs_f32() / span).clamp(0.0, 1.0);
                    return Some((
                        [
                            a.pos[0] + (s.pos[0] - a.pos[0]) * f,
                            a.pos[1] + (s.pos[1] - a.pos[1]) * f,
                            a.pos[2] + (s.pos[2] - a.pos[2]) * f,
                        ],
                        lerp_angle(a.yaw, s.yaw, f),
                        a.pitch + (s.pitch - a.pitch) * f,
                    ));
                }
                // target is before our oldest sample — use the oldest.
                return Some((s.pos, s.yaw, s.pitch));
            }
            prev = Some(s);
        }
        // target is past our newest sample (stall) — hold the latest pose.
        self.samples.back().map(|s| (s.pos, s.yaw, s.pitch))
    }
}

impl Default for RemotePlayer {
    fn default() -> Self {
        Self::new()
    }
}

fn lerp_angle(a: f32, b: f32, f: f32) -> f32 {
    let mut d = b - a;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    while d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    a + d * f
}

/// UDP pose datagram. `token` authenticates the sender (from JoinAck) so the
/// server can map an unauthenticated UDP packet back to a player.
#[derive(Serialize, Deserialize)]
struct PosePacket {
    token: u64,
    id: PlayerId,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

// ---------------- client ----------------

pub struct NetClient {
    out: Sender<Message>,
    inbox: Receiver<Message>,
    udp: Arc<UdpSocket>,
    server_udp: Mutex<Option<SocketAddr>>,
    token: std::sync::atomic::AtomicU64,
    /// Cleared when the TCP reader/writer thread dies — the host uses this to
    /// trigger a reconnect (checklist: reconnection).
    connected: Arc<AtomicBool>,
    /// Set in Drop so the worker threads exit instead of leaking on reconnect.
    shutdown: Arc<AtomicBool>,
    /// A clone of the TCP stream kept solely to shut it down on Drop, which
    /// unblocks the reader thread's blocking read.
    tcp: TcpStream,
    pub my_id: Option<PlayerId>,
    pub seed: Option<u64>,
}

impl Drop for NetClient {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Unblock the TCP reader (and writer) so their threads exit promptly;
        // the UDP reader wakes from its read timeout and sees `shutdown`.
        let _ = self.tcp.shutdown(std::net::Shutdown::Both);
    }
}

impl NetClient {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        let server_ip = stream.peer_addr()?.ip();
        let stream_r = stream.try_clone()?;
        let stream_ctl = stream.try_clone()?; // kept for shutdown on Drop
        let stream_w = stream;
        let (tx_inbound, rx_inbound) = unbounded::<Message>();
        let (tx_outbound, rx_outbound) = unbounded::<Message>();

        // UDP socket for pose (ephemeral local port). A read timeout lets the
        // reader thread periodically check the shutdown flag instead of blocking
        // on recv_from forever (which would leak the thread on reconnect).
        let udp = Arc::new(UdpSocket::bind(("0.0.0.0", 0))?);
        udp.set_read_timeout(Some(Duration::from_secs(1)))?;
        let connected = Arc::new(AtomicBool::new(true));
        let shutdown = Arc::new(AtomicBool::new(false));

        let tx_in_tcp = tx_inbound.clone();
        let connected_r = connected.clone();
        thread::spawn(move || {
            let mut r = std::io::BufReader::new(stream_r);
            loop {
                match read_message(&mut r) {
                    Ok(m) => {
                        if tx_in_tcp.send(m).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            connected_r.store(false, Ordering::Relaxed);
        });
        let connected_w = connected.clone();
        thread::spawn(move || {
            let mut w = std::io::BufWriter::new(stream_w);
            for m in rx_outbound.iter() {
                if write_message(&mut w, &m).is_err() {
                    break;
                }
            }
            connected_w.store(false, Ordering::Relaxed);
        });

        // UDP reader: surface incoming poses into the same inbox.
        let udp_r = udp.clone();
        let tx_in_udp = tx_inbound;
        let shutdown_u = shutdown.clone();
        thread::spawn(move || {
            let mut buf = [0u8; 512];
            loop {
                if shutdown_u.load(Ordering::Relaxed) {
                    break;
                }
                match udp_r.recv_from(&mut buf) {
                    Ok((n, _src)) => {
                        if let Ok(p) = bincode::deserialize::<PosePacket>(&buf[..n]) {
                            let m = Message::PlayerUpdate { id: p.id, pos: p.pos, yaw: p.yaw, pitch: p.pitch };
                            if tx_in_udp.send(m).is_err() {
                                break;
                            }
                        }
                    }
                    // Read timeout — loop back to check the shutdown flag.
                    Err(ref e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(_) => break,
                }
            }
        });

        let client = NetClient {
            out: tx_outbound,
            inbox: rx_inbound,
            udp,
            server_udp: Mutex::new(None),
            token: std::sync::atomic::AtomicU64::new(0),
            connected,
            shutdown,
            tcp: stream_ctl,
            my_id: None,
            seed: None,
        };
        client.out.send(Message::Hello { version: PROTOCOL_VERSION }).ok();
        // Remember the server IP so we can target its UDP port from JoinAck.
        *client.server_udp.lock().unwrap() = Some(SocketAddr::new(server_ip, 0));
        Ok(client)
    }

    /// Reliable send (TCP) — edits, control, heartbeat.
    pub fn send(&self, msg: Message) {
        let _ = self.out.send(msg);
    }

    /// Best-effort pose send (UDP).
    pub fn send_pose(&self, id: PlayerId, pos: [f32; 3], yaw: f32, pitch: f32) {
        let token = self.token.load(Ordering::Relaxed);
        let dst = { *self.server_udp.lock().unwrap() };
        let Some(dst) = dst else { return; };
        if dst.port() == 0 {
            return; // JoinAck not processed yet
        }
        let pkt = PosePacket { token, id, pos, yaw, pitch };
        if let Ok(bytes) = bincode::serialize(&pkt) {
            let _ = self.udp.send_to(&bytes, dst);
        }
    }

    pub fn heartbeat(&self) {
        let _ = self.out.send(Message::Heartbeat);
    }

    /// False once the TCP link has dropped; the host then reconnects.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Drain all buffered messages, applying JoinAck side effects (id/seed/token
    /// + server UDP endpoint).
    pub fn drain(&mut self) -> Vec<Message> {
        let mut out = Vec::new();
        loop {
            match self.inbox.try_recv() {
                Ok(m) => {
                    match &m {
                        Message::JoinAck { your_id, seed, token, udp_port } => {
                            self.my_id = Some(*your_id);
                            self.seed = Some(*seed);
                            self.token.store(*token, Ordering::Relaxed);
                            if let Some(mut g) = self.server_udp.lock().ok() {
                                if let Some(addr) = g.as_mut() {
                                    addr.set_port(*udp_port);
                                }
                            }
                        }
                        Message::VersionMismatch { server_version } => {
                            log::error!(
                                "server protocol v{} != client v{} — update required",
                                server_version, PROTOCOL_VERSION
                            );
                        }
                        _ => {}
                    }
                    out.push(m);
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return out,
            }
        }
    }
}

// ---------------- server ----------------

struct NewClient {
    id: PlayerId,
    tx: Sender<Message>,
}

pub struct NetServer {
    inbound: Receiver<(PlayerId, Message)>,
    new_clients: Receiver<NewClient>,
    clients: HashMap<PlayerId, Sender<Message>>,
    last_seen: HashMap<PlayerId, Instant>,
    /// token → id, shared with the UDP reader thread.
    tokens: Arc<Mutex<HashMap<u64, PlayerId>>>,
    /// id → learned UDP address (filled when the client's first pose arrives).
    udp_addrs: Arc<Mutex<HashMap<PlayerId, SocketAddr>>>,
    udp: Arc<UdpSocket>,
    udp_port: u16,
    next_token: u64,
}

impl NetServer {
    pub fn listen(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let udp = Arc::new(UdpSocket::bind(("0.0.0.0", port))?);
        log::info!("server listening on 0.0.0.0:{} (tcp+udp)", port);
        let next_id = Arc::new(AtomicU32::new(1));
        let (tx_inbound, rx_inbound) = unbounded::<(PlayerId, Message)>();
        let (tx_new, rx_new) = unbounded::<NewClient>();
        let tokens: Arc<Mutex<HashMap<u64, PlayerId>>> = Arc::new(Mutex::new(HashMap::new()));
        let udp_addrs: Arc<Mutex<HashMap<PlayerId, SocketAddr>>> = Arc::new(Mutex::new(HashMap::new()));

        // TCP accept loop.
        {
            let tx_inbound = tx_inbound.clone();
            thread::spawn(move || {
                for stream_res in listener.incoming() {
                    let Ok(stream) = stream_res else { continue; };
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let _ = stream.set_nodelay(true);
                    let Ok(stream_r) = stream.try_clone() else { continue; };
                    let stream_w = stream;
                    let (tx_out, rx_out) = unbounded::<Message>();

                    let tx_in_clone = tx_inbound.clone();
                    let tx_new_clone = tx_new.clone();
                    let tx_out_for_new = tx_out.clone();
                    thread::spawn(move || {
                        let mut r = std::io::BufReader::new(stream_r);
                        // Gate the join on a valid Hello with a matching version.
                        match read_message(&mut r) {
                            Ok(Message::Hello { version }) if version == PROTOCOL_VERSION => {}
                            Ok(Message::Hello { version }) => {
                                let _ = tx_out_for_new.send(Message::VersionMismatch {
                                    server_version: PROTOCOL_VERSION,
                                });
                                log::warn!("rejected client {} (protocol v{} != v{})", id, version, PROTOCOL_VERSION);
                                // Give the writer a moment to flush, then drop.
                                thread::sleep(Duration::from_millis(100));
                                return;
                            }
                            _ => return,
                        }
                        // The UDP auth token is issued lazily in issue_token()
                        // (sent in JoinAck) — no token is registered here.
                        let _ = tx_new_clone.send(NewClient { id, tx: tx_out_for_new });
                        loop {
                            match read_message(&mut r) {
                                Ok(m) => {
                                    if tx_in_clone.send((id, m)).is_err() {
                                        break;
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                        let _ = tx_in_clone.send((id, Message::PlayerLeave { id }));
                    });
                    thread::spawn(move || {
                        let mut w = std::io::BufWriter::new(stream_w);
                        for m in rx_out.iter() {
                            if write_message(&mut w, &m).is_err() {
                                break;
                            }
                        }
                    });
                    log::info!("client {} connected", id);
                }
            });
        }

        // UDP receive loop: authenticate by token, learn the source address,
        // surface the pose as a normal inbound message.
        {
            let udp_r = udp.clone();
            let tokens = tokens.clone();
            let udp_addrs = udp_addrs.clone();
            let tx_inbound = tx_inbound;
            thread::spawn(move || {
                let mut buf = [0u8; 512];
                loop {
                    match udp_r.recv_from(&mut buf) {
                        Ok((n, src)) => {
                            let Ok(p) = bincode::deserialize::<PosePacket>(&buf[..n]) else { continue; };
                            let id = { tokens.lock().unwrap().get(&p.token).copied() };
                            let Some(id) = id else { continue; };
                            udp_addrs.lock().unwrap().insert(id, src);
                            let m = Message::PlayerUpdate { id, pos: p.pos, yaw: p.yaw, pitch: p.pitch };
                            if tx_inbound.send((id, m)).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        Ok(NetServer {
            inbound: rx_inbound,
            new_clients: rx_new,
            clients: HashMap::new(),
            last_seen: HashMap::new(),
            tokens,
            udp_addrs,
            udp,
            udp_port: port,
            next_token: 1,
        })
    }

    pub fn udp_port(&self) -> u16 {
        self.udp_port
    }

    /// Issue a UDP auth token for a freshly joined client (registers it for the
    /// UDP reader thread). Returns the token to put in JoinAck.
    pub fn issue_token(&mut self, id: PlayerId) -> u64 {
        let token = self.next_token;
        self.next_token = self.next_token.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.tokens.lock().unwrap().insert(token, id);
        token
    }

    /// Poll new connections + inbound messages, and time out silent clients.
    /// Returns (joined ids, messages, timed-out ids).
    pub fn poll(&mut self) -> (Vec<PlayerId>, Vec<(PlayerId, Message)>, Vec<PlayerId>) {
        let mut joined = Vec::new();
        while let Ok(nc) = self.new_clients.try_recv() {
            self.clients.insert(nc.id, nc.tx);
            self.last_seen.insert(nc.id, Instant::now());
            joined.push(nc.id);
        }
        let mut msgs = Vec::new();
        while let Ok((id, m)) = self.inbound.try_recv() {
            self.last_seen.insert(id, Instant::now());
            msgs.push((id, m));
        }
        // Heartbeat timeout sweep.
        let now = Instant::now();
        let timed_out: Vec<PlayerId> = self
            .last_seen
            .iter()
            .filter(|(_, &t)| now.duration_since(t) > CLIENT_TIMEOUT)
            .map(|(&id, _)| id)
            .collect();
        for &id in &timed_out {
            log::info!("server: player {} timed out", id);
            self.drop_client(id);
        }
        (joined, msgs, timed_out)
    }

    pub fn send_to(&self, id: PlayerId, msg: Message) {
        if let Some(tx) = self.clients.get(&id) {
            let _ = tx.send(msg);
        }
    }

    pub fn broadcast(&self, msg: &Message, except: Option<PlayerId>) {
        for (id, tx) in &self.clients {
            if Some(*id) == except {
                continue;
            }
            let _ = tx.send(msg.clone());
        }
    }

    pub fn broadcast_filter<F: FnMut(PlayerId) -> bool>(&self, msg: &Message, mut keep: F) {
        for (id, tx) in &self.clients {
            if keep(*id) {
                let _ = tx.send(msg.clone());
            }
        }
    }

    /// Fan a pose out over UDP to clients matching the predicate (interest
    /// management). Clients whose UDP address isn't learned yet are skipped.
    pub fn broadcast_pose_filter<F: FnMut(PlayerId) -> bool>(
        &self,
        id: PlayerId,
        pos: [f32; 3],
        yaw: f32,
        pitch: f32,
        mut keep: F,
    ) {
        let pkt = PosePacket { token: 0, id, pos, yaw, pitch };
        let Ok(bytes) = bincode::serialize(&pkt) else { return; };
        let addrs = self.udp_addrs.lock().unwrap();
        for (&cid, &addr) in addrs.iter() {
            if cid != id && keep(cid) {
                let _ = self.udp.send_to(&bytes, addr);
            }
        }
    }

    pub fn drop_client(&mut self, id: PlayerId) {
        self.clients.remove(&id);
        self.last_seen.remove(&id);
        self.udp_addrs.lock().unwrap().remove(&id);
        self.tokens.lock().unwrap().retain(|_, &mut v| v != id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_log_compress_roundtrip() {
        let edits: Vec<Message> = (0..1000)
            .map(|i| Message::VoxelEdit { wx: i, wy: i * 2, wz: -i, mat: (i % 32) as u8, seq: i as u64 })
            .collect();
        let comp = compress_edits(&edits);
        // Highly repetitive → should compress well.
        assert!(comp.len() < bincode::serialize(&edits).unwrap().len());
        let back = decompress_edits(&comp);
        assert_eq!(back.len(), edits.len());
        match (&edits[42], &back[42]) {
            (Message::VoxelEdit { wx: a, seq: sa, .. }, Message::VoxelEdit { wx: b, seq: sb, .. }) => {
                assert_eq!(a, b);
                assert_eq!(sa, sb);
            }
            _ => panic!("variant changed"),
        }
    }

    #[test]
    fn interpolation_blends_between_samples() {
        let mut rp = RemotePlayer::new();
        let t0 = Instant::now();
        // Two samples 200ms apart; query at the midpoint-in-the-past.
        rp.push_sample(t0, [0.0, 0.0, 0.0], 0.0, 0.0);
        rp.push_sample(t0 + Duration::from_millis(200), [10.0, 0.0, 0.0], 0.0, 0.0);
        // now is 100ms after the 2nd sample → target = now-100ms = 2nd sample time.
        let now = t0 + Duration::from_millis(300);
        let (pos, _, _) = rp.sample(now).expect("has samples");
        // target == t0+200ms → should be at the 2nd sample (x≈10).
        assert!((pos[0] - 10.0).abs() < 0.01, "got {pos:?}");

        // Query earlier: now = t0+200ms → target = t0+100ms → halfway → x≈5.
        let (pos2, _, _) = rp.sample(t0 + Duration::from_millis(200)).expect("has samples");
        assert!((pos2[0] - 5.0).abs() < 0.5, "expected ~5 got {pos2:?}");
    }

    #[test]
    fn version_constant_is_stable() {
        // Guards against accidental bumps without intent.
        assert_eq!(PROTOCOL_VERSION, 1);
    }
}
