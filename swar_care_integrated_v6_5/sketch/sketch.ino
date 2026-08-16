#include <Arduino_RouterBridge.h>
#include <zephyr/kernel.h>
#include <string.h>

// ============================================================================
// UNO Q STM32 MCU MASTER-SYNC PIEZO STREAM
// IMPORTANT: acquisition rates are unchanged.
//   Piezo = 2000 Hz
//   Batch = 40 samples = 20 ms
// USB microphone remains 16000 Hz on the MPU side.
// ============================================================================

const int PIEZO_PIN      = A0;
const int SAMPLE_RATE_HZ = 2000;
const int BATCH_SIZE     = 40;
const uint32_t SYNC_EVERY_SAMPLES = 2000; // 1 second

struct __attribute__((packed)) TelemetryPacket {
  uint32_t batch_start_idx;
  uint64_t batch_start_us;
  uint16_t samples[BATCH_SIZE];
};

static_assert(sizeof(TelemetryPacket) == 92, "TelemetryPacket must be exactly 92 bytes for MPU engine unpacking");

volatile uint16_t bufA[BATCH_SIZE];
volatile uint16_t bufB[BATCH_SIZE];
volatile uint16_t* currentBuf = bufA;
volatile uint16_t* sendBuf = nullptr;
volatile int bufIndex = 0;

// These are recording-relative counters. They start at zero at MASTER_START.
volatile uint32_t recordingSampleCount = 0;
volatile uint32_t currentBatchStart = 0;
volatile uint64_t currentBatchStartUs = 0;
volatile uint32_t readyBatchStart = 0;
volatile uint64_t readyBatchStartUs = 0;
volatile bool batchReady = false;

// RPC requests are only flags. Bridge.provide() runs on a high-priority RPC
// thread, so it must remain short and must not call Bridge/Monitor itself.
volatile bool armPending = false;
volatile bool stopPending = false;
volatile uint32_t requestedSessionId = 0;
volatile bool recordingActive = false;
volatile uint32_t activeSessionId = 0;

// Main-loop event flags. Exact timestamps/sample indices are captured by the
// sampler thread; Bridge notifications are sent later from loop().
volatile bool startEventPending = false;
volatile uint32_t startEventSessionId = 0;
volatile uint32_t startEventSample = 0;
volatile uint64_t startEventUs = 0;

volatile bool stopEventPending = false;
volatile uint32_t stopEventSessionId = 0;
volatile uint32_t stopEventSample = 0;
volatile uint64_t stopEventUs = 0;

volatile bool syncEventPending = false;
volatile uint32_t syncEventSessionId = 0;
volatile uint32_t syncEventSample = 0;
volatile uint64_t syncEventUs = 0;
volatile uint32_t nextSyncSample = SYNC_EVERY_SAMPLES;

struct k_timer sample_timer;
struct k_thread sampler_thread;
K_THREAD_STACK_DEFINE(sampler_stack, 2048);

// ---------------------------------------------------------------------------
// MPU -> MCU RPC controls
// ---------------------------------------------------------------------------

void arm_recording(uint32_t sessionId) {
  if (recordingActive) return;
  requestedSessionId = sessionId;
  armPending = true;
}

void stop_recording(uint32_t sessionId) {
  if (!recordingActive) return;
  if (sessionId != activeSessionId) return;
  stopPending = true;
}

// ---------------------------------------------------------------------------
// Exact 2 kHz sampler
// ---------------------------------------------------------------------------

void sampler_thread_entry(void *p1, void *p2, void *p3) {
  k_timer_init(&sample_timer, NULL, NULL);
  k_timer_start(&sample_timer, K_USEC(500), K_USEC(500));

  while (1) {
    k_timer_status_sync(&sample_timer);

    // START is user-controlled from the MPU, but the actual epoch is created
    // here at an exact sampler boundary. Any partial idle batch is discarded.
    if (armPending && !recordingActive) {
      armPending = false;
      activeSessionId = requestedSessionId;
      recordingActive = true;

      bufIndex = 0;
      recordingSampleCount = 0;
      currentBatchStart = 0;
      currentBatchStartUs = k_ticks_to_us_near64(k_uptime_ticks());
      nextSyncSample = SYNC_EVERY_SAMPLES;

      startEventSessionId = activeSessionId;
      startEventSample = 0;
      startEventUs = currentBatchStartUs;
      startEventPending = true;
    }

    if (!recordingActive) {
      continue;
    }

    // STOP is honored only at a 40-sample boundary. Therefore the final
    // recording endpoint is deterministic and exactly representable in both
    // rates: 40 piezo samples = 320 nominal 16-kHz audio samples.
    if (bufIndex == 0) {
      currentBatchStart = recordingSampleCount;
      currentBatchStartUs = k_ticks_to_us_near64(k_uptime_ticks());
    }

    currentBuf[bufIndex++] = (uint16_t)analogRead(PIEZO_PIN);
    recordingSampleCount++;

    if (bufIndex >= BATCH_SIZE) {
      sendBuf = currentBuf;
      readyBatchStart = currentBatchStart;
      readyBatchStartUs = currentBatchStartUs;
      currentBuf = (currentBuf == bufA) ? bufB : bufA;
      bufIndex = 0;
      batchReady = true;

      if (recordingSampleCount >= nextSyncSample) {
        syncEventSessionId = activeSessionId;
        syncEventSample = recordingSampleCount;
        syncEventUs = k_ticks_to_us_near64(k_uptime_ticks());
        syncEventPending = true;
        nextSyncSample += SYNC_EVERY_SAMPLES;
      }

      if (stopPending) {
        stopPending = false;
        recordingActive = false;

        stopEventSessionId = activeSessionId;
        stopEventSample = recordingSampleCount;
        stopEventUs = k_ticks_to_us_near64(k_uptime_ticks());
        stopEventPending = true;
      }
    }
  }
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  analogReadResolution(12);
  pinMode(PIEZO_PIN, INPUT);

  // Python calls these RPC services from the START/STOP buttons.
  Bridge.provide("arm_recording", arm_recording);
  Bridge.provide("stop_recording", stop_recording);

  k_thread_create(
    &sampler_thread,
    sampler_stack,
    K_THREAD_STACK_SIZEOF(sampler_stack),
    sampler_thread_entry,
    NULL, NULL, NULL,
    0, 0, K_NO_WAIT
  );

  Monitor.println("Piezo master-clock sync streamer ready");
}

void loop() {
  // Send captured data from the main loop, keeping Bridge calls out of the
  // real-time sampler thread.
  if (batchReady) {
    noInterrupts();
    volatile uint16_t* bufferToSend = sendBuf;
    uint32_t batchStartToSend = readyBatchStart;
    uint64_t batchStartUsToSend = readyBatchStartUs;
    batchReady = false;
    interrupts();

    sendBatch(bufferToSend, batchStartToSend, batchStartUsToSend);
  }

  if (startEventPending) {
    noInterrupts();
    uint32_t sid = startEventSessionId;
    uint32_t sample = startEventSample;
    uint64_t us = startEventUs;
    startEventPending = false;
    interrupts();
    Bridge.notify("mcu_recording_started", sid, sample, us);
  }

  if (syncEventPending) {
    noInterrupts();
    uint32_t sid = syncEventSessionId;
    uint32_t sample = syncEventSample;
    uint64_t us = syncEventUs;
    syncEventPending = false;
    interrupts();
    Bridge.notify("mcu_sync_marker", sid, sample, us);
  }

  if (stopEventPending) {
    noInterrupts();
    uint32_t sid = stopEventSessionId;
    uint32_t sample = stopEventSample;
    uint64_t us = stopEventUs;
    stopEventPending = false;
    interrupts();
    Bridge.notify("mcu_recording_stopped", sid, sample, us);
  }

  k_yield();
}

void sendBatch(volatile uint16_t* buffer, uint32_t startIndex, uint64_t startUs) {
  TelemetryPacket packet;
  packet.batch_start_idx = startIndex;
  packet.batch_start_us = startUs;

  for (int i = 0; i < BATCH_SIZE; i++) {
    packet.samples[i] = buffer[i];
  }

  MsgPack::bin_t<unsigned char> msgpack_bin_blob;
  msgpack_bin_blob.resize(sizeof(TelemetryPacket));
  memcpy(msgpack_bin_blob.data(), &packet, sizeof(TelemetryPacket));

  Bridge.notify("piezo_batch", msgpack_bin_blob);
}