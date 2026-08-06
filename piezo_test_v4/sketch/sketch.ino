#include <Arduino_RouterBridge.h>
#include <zephyr/kernel.h>

const int PIEZO_PIN      = A0;
const int SAMPLE_RATE_HZ = 2000;
const int BATCH_SIZE     = 40;

// ==============================================================================
// 1. CHRONO-ANCHORED BINARY STRUCTURE DEFINITION
// ==============================================================================
struct __attribute__((packed)) TelemetryPacket {
  uint32_t batch_start_idx;       // 4 bytes
  uint64_t batch_start_us;        // 8 bytes -> Zephyr high-res microsecond clock
  uint16_t samples[BATCH_SIZE];   // 80 bytes (40 samples * 2 bytes)
}; // Total structure size: Exactly 92 bytes

volatile uint16_t bufA[BATCH_SIZE];
volatile uint16_t bufB[BATCH_SIZE];
volatile uint16_t* currentBuf = bufA;
volatile uint16_t* sendBuf = nullptr;
volatile int bufIndex = 0;
volatile bool batchReady = false;
volatile uint32_t absoluteSampleCount = 0;

volatile uint32_t currentBatchStart = 0;
volatile uint32_t readyBatchStart = 0;

volatile uint64_t currentBatchStartUs = 0;
volatile uint64_t readyBatchStartUs = 0;

struct k_timer sample_timer;
struct k_thread sampler_thread;
K_THREAD_STACK_DEFINE(sampler_stack, 1024);

void sampler_thread_entry(void *p1, void *p2, void *p3) {
  k_timer_init(&sample_timer, NULL, NULL);
  k_timer_start(&sample_timer, K_USEC(500), K_USEC(500));
  
  while (1) {
    k_timer_status_sync(&sample_timer);
    
    if (bufIndex == 0) {
      currentBatchStart = absoluteSampleCount;
      currentBatchStartUs = k_ticks_to_us_near64(k_uptime_ticks());
    }
    
    currentBuf[bufIndex++] = (uint16_t)analogRead(PIEZO_PIN);
    absoluteSampleCount++;
    
    if (bufIndex >= BATCH_SIZE) {
      sendBuf = currentBuf;
      readyBatchStart = currentBatchStart; 
      readyBatchStartUs = currentBatchStartUs;
      currentBuf = (currentBuf == bufA) ? bufB : bufA;
      bufIndex = 0;
      batchReady = true;
    }
  }
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  analogReadResolution(12);
  pinMode(PIEZO_PIN, INPUT);
  
  k_thread_create(&sampler_thread, sampler_stack, K_THREAD_STACK_SIZEOF(sampler_stack),
                  sampler_thread_entry, NULL, NULL, NULL,
                  1, 0, K_NO_WAIT);
                  
  Monitor.print("Piezo streaming started - Microsecond Exact Hardware Timing Mode");
}

void loop() {
  if (batchReady) {
    noInterrupts();
    volatile uint16_t* bufferToSend = sendBuf;
    uint32_t batchStartToSend = readyBatchStart;
    uint64_t batchStartUsToSend = readyBatchStartUs;
    batchReady = false;
    interrupts();
    
    sendBatch(bufferToSend, batchStartToSend, batchStartUsToSend);
  }
  k_yield();
}

// ==============================================================================
// 2. FIXED: COMPILER COMPATIBLE ARDUINO MSGPACK BINARY WRAPPING
// ==============================================================================
void sendBatch(volatile uint16_t* buffer, uint32_t startIndex, uint64_t startUs) {
  TelemetryPacket packet;
  packet.batch_start_idx = startIndex;
  packet.batch_start_us = startUs;
  
  for (int i = 0; i < BATCH_SIZE; i++) {
    packet.samples[i] = buffer[i];
  }
  
  // Create an explicit binary buffer array type recognized by hideakitai's MsgPack
  MsgPack::bin_t<unsigned char> msgpack_bin_blob;
  
  // Resize the container to perfectly house the 92 bytes of the packed struct
  msgpack_bin_blob.resize(sizeof(TelemetryPacket));
  //Monitor.print("Timestamp: ");
  //Monitor.println((unsigned long long)startUs);
  // Directly copy our tightly packed raw memory blocks into the blob array pointer
  memcpy(msgpack_bin_blob.data(), &packet, sizeof(TelemetryPacket));
  
  // Broadcast across the RouterBridge layout cleanly
  Bridge.notify("piezo_batch", msgpack_bin_blob);
}