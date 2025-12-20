#include <Arduino_HS300x.h>
#include <Arduino_LPS22HB.h>
#include <TensorFlowLite.h>

// TFLite headers
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h" // your int8 TFLite model

// ---------- Normalization parameters ----------
const float x_mean[3] = {6.229321, 27.827852, 1003.1014}; 
const float x_std[3]  = {1.1341126, 3.167768, 0.09006663};
const float y_mean    = 5.1984158;
const float y_std     = 0.9711739;

// ---------- Rolling window ----------
constexpr int WINDOW_SIZE = 15; // 30 min history at 10s sampling
float window[WINDOW_SIZE][3]; 
int window_index = 0;
bool window_filled = false;

// ---------- TFLite setup ----------
constexpr int kTensorArenaSize = 12 * 1024; 
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model_ptr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ---------- Timing ----------
unsigned long lastTime = 0;
const unsigned long interval = 10000; // 10s

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!HS300x.begin()) Serial.println("HS300x init failed!");
  if (!BARO.begin()) Serial.println("BARO init failed!");

  // --- TFLite initialization ---
  model_ptr = tflite::GetModel(model_int8_tflite);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while(1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model_ptr, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while(1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Time(ms),Temp(C),Hum(%),Pres(hPa),Pred_Temp");
}

void loop() {
  if (millis() - lastTime >= interval) {
    lastTime = millis();

    float temp = HS300x.readTemperature();
    float hum  = HS300x.readHumidity();
    float pres = BARO.readPressure() * 10; // hPa

    // --- Update rolling window ---
    window[window_index][0] = temp;
    window[window_index][1] = hum;
    window[window_index][2] = pres;
    window_index = (window_index + 1) % WINDOW_SIZE;
    if (window_index == 0) window_filled = true;

    float predicted_temp = NAN;

    if (window_filled) {
      // --- Fill input tensor ---
      for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (window_index + i) % WINDOW_SIZE; // chronological
        for (int j = 0; j < 3; j++) {
          float norm = (window[idx][j] - x_mean[j]) / x_std[j];
          input->data.int8[i * 3 + j] =
              (int8_t)(norm / input->params.scale + input->params.zero_point);
        }
      }

      // --- Run inference ---
      if (interpreter->Invoke() == kTfLiteOk) {
        float y_norm = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
        predicted_temp = y_norm * y_std + y_mean;
      }
    }

    // --- Print results ---
    Serial.print(millis()); Serial.print(",");
    Serial.print(temp);     Serial.print(",");
    Serial.print(hum);      Serial.print(",");
    Serial.print(pres);     Serial.print(",");
    if (!isnan(predicted_temp)) Serial.println(predicted_temp);
    else Serial.println("WAITING_FOR_WINDOW");
  }
}