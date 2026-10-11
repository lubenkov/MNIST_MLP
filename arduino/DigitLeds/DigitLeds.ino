const int NUM_LEDS = 9;
const int ledPins[NUM_LEDS] = { 2, 3, 4, 5, 6, 7, 8, 9, 10 };

String inputBuffer = "";

void setLedCount(int count) {
  count = constrain(count, 0, NUM_LEDS);
  for (int i = 0; i < NUM_LEDS; ++i) {
    digitalWrite(ledPins[i], i < count ? HIGH : LOW);
  }
}

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < NUM_LEDS; ++i) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }

  for (int i = 0; i < NUM_LEDS; ++i) {
    digitalWrite(ledPins[i], HIGH);
    delay(40);
  }
  for (int i = 0; i < NUM_LEDS; ++i) {
    digitalWrite(ledPins[i], LOW);
    delay(40);
  }
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      if (inputBuffer.length() > 0) {
        int digit = inputBuffer.toInt();
        setLedCount(digit);
        inputBuffer = "";
      }
    }
    else if (c != '\r') {
      inputBuffer += c;
    }
  }
}
