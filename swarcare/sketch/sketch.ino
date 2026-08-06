/*
 * SwarCare — MCU Sketch (Arduino UNO Q)
 * ========================================
 * Runs on STM32U585 MCU (Cortex-M33, Zephyr RTOS).
 * Provides LED Matrix feedback based on veena health status
 * received from the MPU Python app via Bridge.
 *
 * LED MATRIX (13x8 on UNO Q):
 *   0 = Healthy  → Veena icon (steady)
 *   1 = Watch    → Exclamation mark pulse (1Hz)
 *   2 = Anomaly  → X mark / alert triangle (3Hz flash)
 *
 * Frame format: const uint32_t[4] — 104 pixels (8×13) packed MSB-first
 * into 128 bits, matching air quality / heart examples.
 *
 * BRIDGE API:
 *   Python → MCU: Bridge.notify("set_status", status_code)
 */

#include <Arduino.h>
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

ArduinoLEDMatrix matrix;

// Current veena health status (0=healthy, 1=watch, 2=anomaly)
volatile int currentStatus = 0;
unsigned long previousMillis = 0;
int animFrame = 0;
int lastDisplayedStatus = -1;

// ─── LED Matrix Frames (8 rows × 13 cols, packed uint32_t[4]) ───
// Pixel at [row][col] → flat = row*13+col → word[flat/32] bit (31 - flat%32)

// HEALTHY: Veena icon (stylized instrument)
// .....XX......
// ....X..X.....
// ....X........
// ....X........
// ....X........
// ..XXXXXX.....
// .X..X...X....
// ..XXXXXX.....
const uint32_t frame_veena[4] = {
    0x06004802,
    0x00100080,
    0x1F812207,
    0xE0000000,
};

// WATCH: Exclamation mark !
// .....XX......
// .....XX......
// .....XX......
// .....XX......
// .....XX......
// .............
// .....XX......
// .....XX......
const uint32_t frame_watch_on[4] = {
    0x06003001,
    0x800C0060,
    0x00001800,
    0xC0000000,
};

// Blank (for pulsing animation)
const uint32_t frame_blank[4] = {
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
};

// ANOMALY: X mark
// X..........X.
// .X........X..
// ..X......X...
// ...X....X....
// ....X..X.....
// ...X....X....
// ..X......X...
// XX........XX.
const uint32_t frame_anomaly_x[4] = {
    0x80120108,
    0x10210090,
    0x08408118,
    0x06000000,
};

// ANOMALY: Alert triangle with ! inside
// .....XX......
// ....X..X.....
// ...X....X....
// ...X.XX.X....
// ..X..XX..X...
// .X........X..
// .X...XX...X..
// XXXXXXXXXXXX.
const uint32_t frame_anomaly_tri[4] = {
    0x06004804,
    0x202D0264,
    0x2011189F,
    0xFE000000,
};

/**
 * Bridge handler: Called by Python via Bridge.notify("set_status", code)
 */
void setStatus(int status) {
    currentStatus = status;
    animFrame = 0;
    lastDisplayedStatus = -1;  // Force redraw on status change
}

void setup() {
    Bridge.begin();
    Monitor.begin();

    // Initialize LED Matrix
    matrix.begin();

    // Register Bridge handler
    Bridge.provide("set_status", setStatus);

    // Show startup veena icon
    matrix.loadFrame(frame_veena);
    delay(1000);

    Monitor.println("SwarCare MCU ready — LED Matrix active.");
}

void loop() {
    unsigned long currentMillis = millis();

    switch (currentStatus) {
        case 0:  // HEALTHY — veena icon (display once, hold steady)
            if (lastDisplayedStatus != 0) {
                matrix.loadFrame(frame_veena);
                lastDisplayedStatus = 0;
            }
            break;

        case 1:  // WATCH — exclamation mark pulsing (1Hz)
            if (currentMillis - previousMillis >= 500) {
                previousMillis = currentMillis;
                animFrame = !animFrame;
                if (animFrame) {
                    matrix.loadFrame(frame_watch_on);
                } else {
                    matrix.loadFrame(frame_blank);
                }
                lastDisplayedStatus = 1;
            }
            break;

        case 2:  // ANOMALY — alternating X and alert triangle (3Hz)
            if (currentMillis - previousMillis >= 333) {
                previousMillis = currentMillis;
                animFrame = !animFrame;
                if (animFrame) {
                    matrix.loadFrame(frame_anomaly_x);
                } else {
                    matrix.loadFrame(frame_anomaly_tri);
                }
                lastDisplayedStatus = 2;
            }
            break;
    }

    delay(50);
}
