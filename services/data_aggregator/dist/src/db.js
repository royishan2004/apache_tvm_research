import { drizzle } from "drizzle-orm/node-postgres";
import { config } from "dotenv";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Pool } from "pg";
config({ path: ".env" });
const DEFAULT_IDLE_DISCONNECT_MS = 5 * 60 * 1000;
const DEFAULT_ACTIVITY_CHECK_MS = 10 * 1000;
const TEST_PROCESS_GREP_PATTERN = "qkv_mlp_run|metaschedule_tune|import_bert_matmul_results";
const TEST_PROCESS_PATTERNS = [
    /research\.workloads\.bert\.matmul\.qkv_mlp_run/,
    /research\.workloads\.bert\.metaschedule\.metaschedule_tune/,
    /\bscripts\/import_bert_matmul_results\.py\b/,
    /\bimport_bert_matmul_results\.py\b/,
    /\bqkv_mlp_run\.py\b/,
    /\bmetaschedule_tune\.py\b/,
];
const execFileAsync = promisify(execFile);
let pool = null;
let db = null;
let monitorTimer = null;
let lastActiveTestProcessAt = Date.now();
let shutdownScheduled = false;
let hadActiveTestProcess = false;
function istTimestamp() {
    return `${new Date().toLocaleString("en-GB", { timeZone: "Asia/Kolkata" })} IST`;
}
function getDatabaseUrl() {
    const dbUrl = process.env.DATABASE_URL;
    if (!dbUrl) {
        throw new Error("DATABASE_URL is not set. Start the server with a valid DB URL.");
    }
    return dbUrl;
}
function getIdleDisconnectMs() {
    const raw = Number(process.env.TEST_IDLE_DISCONNECT_MS);
    if (Number.isFinite(raw) && raw >= 1000) {
        return raw;
    }
    return DEFAULT_IDLE_DISCONNECT_MS;
}
function getActivityCheckMs() {
    const raw = Number(process.env.TEST_ACTIVITY_CHECK_MS);
    if (Number.isFinite(raw) && raw >= 1000) {
        return raw;
    }
    return DEFAULT_ACTIVITY_CHECK_MS;
}
function ensureDb() {
    if (db) {
        return db;
    }
    const poolMaxRaw = Number(process.env.DB_POOL_MAX);
    const poolIdleRaw = Number(process.env.DB_POOL_IDLE_TIMEOUT_MS);
    pool = new Pool({
        connectionString: getDatabaseUrl(),
        max: Number.isFinite(poolMaxRaw) && poolMaxRaw > 0 ? poolMaxRaw : 5,
        idleTimeoutMillis: Number.isFinite(poolIdleRaw) && poolIdleRaw >= 1000 ? poolIdleRaw : 30_000,
    });
    db = drizzle(pool);
    console.log(`[db] Connection established (${istTimestamp()})`);
    return db;
}
export async function initializeDbConnection() {
    try {
        ensureDb();
        if (pool) {
            await pool.query("select 1");
            console.log(`[db] Startup health-check passed (${istTimestamp()})`);
        }
    }
    catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.warn(`[db] Startup connection check failed (${istTimestamp()}): ${msg}`);
    }
}
export async function execute(query) {
    return ensureDb().execute(query);
}
export function hasActiveDbConnection() {
    return pool !== null;
}
export function touchActivity(reason) {
    lastActiveTestProcessAt = Date.now();
    console.log(`[db-monitor] Activity detected (${reason}); inactivity timer reset (${istTimestamp()})`);
}
export async function disconnectDb(reason) {
    if (!pool) {
        return;
    }
    const activePool = pool;
    pool = null;
    db = null;
    try {
        await activePool.end();
        console.log(`[db] Connection closed (${reason}) (${istTimestamp()})`);
    }
    catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.warn(`[db] Failed to close connection cleanly: ${msg}`);
    }
}
function hasActiveTestProcess(processSnapshot) {
    for (const line of processSnapshot.split("\n")) {
        if (!line.trim()) {
            continue;
        }
        if (TEST_PROCESS_PATTERNS.some((pattern) => pattern.test(line))) {
            return true;
        }
    }
    return false;
}
async function detectActiveTestProcess() {
    try {
        const { stdout } = await execFileAsync("pgrep", ["-af", TEST_PROCESS_GREP_PATTERN], { maxBuffer: 512 * 1024 });
        return hasActiveTestProcess(stdout);
    }
    catch (error) {
        // pgrep exits with status 1 when no process matches.
        const code = error.code;
        if (code === 1) {
            return false;
        }
    }
    const { stdout } = await execFileAsync("ps", ["-eo", "command"], {
        maxBuffer: 2 * 1024 * 1024,
    });
    return hasActiveTestProcess(stdout);
}
async function checkTestActivity() {
    try {
        const now = Date.now();
        const isActiveNow = await detectActiveTestProcess();
        if (isActiveNow) {
            hadActiveTestProcess = true;
            lastActiveTestProcessAt = now;
            return;
        }
        if (hadActiveTestProcess) {
            // A tracked test process just finished; restart inactivity timing from now.
            hadActiveTestProcess = false;
            lastActiveTestProcessAt = now;
            console.log(`[db-monitor] Test process finished; inactivity timer reset (${istTimestamp()})`);
            return;
        }
        const idleMs = now - lastActiveTestProcessAt;
        if (idleMs < getIdleDisconnectMs()) {
            return;
        }
        if (shutdownScheduled) {
            return;
        }
        shutdownScheduled = true;
        const idleMinutes = (idleMs / 60_000).toFixed(1);
        if (hasActiveDbConnection()) {
            await disconnectDb(`idle for ${idleMinutes}m with no test process`);
        }
        // After idleness, perform a graceful shutdown so dev servers don't linger.
        try {
            console.log(`[db-monitor] Idle timeout reached; shutting down server (${istTimestamp()})`);
            stopDbIdleMonitor();
        }
        catch (err) {
            // ignore
        }
        // Give logs a moment then exit
        setTimeout(() => {
            console.log('[db-monitor] Exiting process due to idle timeout');
            process.exit(0);
        }, 250);
    }
    catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.warn(`[db-monitor] Could not inspect test process activity: ${msg}`);
    }
}
export function startDbIdleMonitor() {
    if (monitorTimer) {
        return;
    }
    lastActiveTestProcessAt = Date.now();
    console.log(`[db-monitor] Active. Disconnect timeout=${Math.round(getIdleDisconnectMs() / 1000)}s, check interval=${Math.round(getActivityCheckMs() / 1000)}s (${istTimestamp()})`);
    monitorTimer = setInterval(() => {
        void checkTestActivity();
    }, getActivityCheckMs());
    if (typeof monitorTimer.unref === "function") {
        monitorTimer.unref();
    }
    void checkTestActivity();
}
export function stopDbIdleMonitor() {
    if (!monitorTimer) {
        return;
    }
    clearInterval(monitorTimer);
    monitorTimer = null;
}
