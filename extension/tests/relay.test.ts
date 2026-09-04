/**
 * The Chrome service worker's port relay, and specifically the buffer in it.
 *
 * A content script posts `hello` in the same turn it calls `connect()`, while the service
 * worker is still awaiting `chrome.offscreen.createDocument()` -- slow the first time.
 * Chrome does not queue port messages for a listener that does not exist yet, so if the
 * relay attached its listener only after the await, that first `hello` was dropped and the
 * page was simply never scored.
 *
 * It failed silently, only on the first page after a browser start, and a page that is never
 * scored looks exactly like a page with nothing to flag. That is the worst shape a bug can
 * have, and it is why the relay attaches synchronously and buffers.
 */

import { describe, expect, it, vi } from "vitest";

import { relay, type RelayEnv } from "../src/router/chrome.js";

class FakePort {
  readonly posted: unknown[] = [];
  disconnected = false;
  #messageListeners = new Set<(message: unknown) => void>();
  #disconnectListeners = new Set<() => void>();

  readonly onMessage = { addListener: (fn: (m: unknown) => void) => this.#messageListeners.add(fn) };
  readonly onDisconnect = { addListener: (fn: () => void) => this.#disconnectListeners.add(fn) };

  postMessage(message: unknown): void {
    this.posted.push(message);
  }
  /**
   * What the code under test calls. Chrome does NOT run your own onDisconnect listeners
   * here -- it notifies the other end -- and modelling that faithfully matters: the relay
   * wires each port's disconnect to the other's, so a fake that fires locally would
   * ping-pong forever and report a loop that cannot happen in a browser.
   */
  disconnect(): void {
    this.disconnected = true;
  }
  /** The other end went away. */
  close(): void {
    for (const fn of this.#disconnectListeners) fn();
  }
  send(message: unknown): void {
    for (const fn of this.#messageListeners) fn(message);
  }
  as(): chrome.runtime.Port {
    return this as unknown as chrome.runtime.Port;
  }
}

/** A relay env whose offscreen document is created only when the test says so. */
function deferred() {
  const downstream = new FakePort();
  const errors: unknown[] = [];
  let release!: () => void;
  let reject!: (err: unknown) => void;
  const ready = new Promise<void>((res, rej) => {
    release = res;
    reject = rej;
  });
  const env: RelayEnv = {
    ensure: () => ready,
    connectDownstream: () => downstream.as(),
    onError: (err) => void errors.push(err),
  };
  return { env, downstream, errors, release, reject };
}

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

describe("buffering while the offscreen document is being created", () => {
  it("delivers a hello posted before the document exists", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);

    upstream.send({ type: "hello" });
    expect(downstream.posted).toEqual([]); // nowhere to go yet

    release();
    await flush();
    expect(downstream.posted).toEqual([{ type: "hello" }]);
  });

  it("delivers a whole burst, in order", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);

    upstream.send({ type: "hello" });
    upstream.send({ type: "score", items: [{ hash: "a" }] });
    upstream.send({ type: "keepalive" });

    release();
    await flush();
    expect(downstream.posted).toEqual([
      { type: "hello" },
      { type: "score", items: [{ hash: "a" }] },
      { type: "keepalive" },
    ]);
  });

  it("passes messages straight through once the document is up", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);
    release();
    await flush();

    upstream.send({ type: "keepalive" });
    expect(downstream.posted).toEqual([{ type: "keepalive" }]);
  });

  it("does not replay the buffer a second time", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);

    upstream.send({ type: "hello" });
    release();
    await flush();
    upstream.send({ type: "keepalive" });

    expect(downstream.posted).toEqual([{ type: "hello" }, { type: "keepalive" }]);
  });
});

describe("relaying back to the page", () => {
  it("forwards what the host sends", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);
    release();
    await flush();

    downstream.send({ type: "ready", calibration: {} });
    expect(upstream.posted).toEqual([{ type: "ready", calibration: {} }]);
  });

  it("closes the page port when the host port goes away", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);
    release();
    await flush();

    downstream.close();
    expect(upstream.disconnected).toBe(true);
  });

  it("survives a page that has already gone", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    upstream.postMessage = () => {
      throw new Error("page went away");
    };
    relay(upstream.as(), env);
    release();
    await flush();

    expect(() => downstream.send({ type: "ready" })).not.toThrow();
  });
});

describe("teardown", () => {
  it("connects nothing when the page disconnects during creation", async () => {
    const connectDownstream = vi.fn();
    const { env, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), { ...env, connectDownstream });

    upstream.close();
    release();
    await flush();

    expect(connectDownstream).not.toHaveBeenCalled();
  });

  it("closes the host port when the page disconnects afterwards", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);
    release();
    await flush();

    upstream.close();
    expect(downstream.disconnected).toBe(true);
  });

  it("does not bounce the disconnect back and forth", () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);
    release();

    expect(() => {
      upstream.close();
      downstream.close();
    }).not.toThrow();
  });

  it("survives a host port that throws while restarting", async () => {
    const { env, downstream, release } = deferred();
    const upstream = new FakePort();
    downstream.postMessage = () => {
      throw new Error("offscreen restarting");
    };
    relay(upstream.as(), env);
    release();
    await flush();

    expect(() => upstream.send({ type: "keepalive" })).not.toThrow();
  });
});

describe("when the offscreen document cannot be created", () => {
  it("reports the failure and closes the port rather than hanging", async () => {
    const { env, errors, reject } = deferred();
    const upstream = new FakePort();
    relay(upstream.as(), env);

    upstream.send({ type: "hello" });
    reject(new Error("Only a single offscreen document may be created"));
    await flush();

    expect(errors).toHaveLength(1);
    expect(upstream.disconnected).toBe(true);
  });
});
