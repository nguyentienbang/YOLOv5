#include <Wire.h>
#include <MPU6050.h>
#include <TinyGPS++.h>
#include <SoftwareSerial.h>



// ==========================================
// MPU6050
// ==========================================

MPU6050 mpu;



// ==========================================
// GPS
// ==========================================

TinyGPSPlus gps;

SoftwareSerial gpsSerial(4, 3);



// ==========================================
// HC-SR04
// ==========================================

#define TRIG_PIN 9
#define ECHO_PIN 10



// ==========================================
// VARIABLES
// ==========================================

float distance_cm = 0;

int16_t ax, ay, az;
int16_t gx, gy, gz;

double latitude = 0;
double longitude = 0;



// ==========================================
// SETUP
// ==========================================

void setup()
{
    Serial.begin(115200);

    // ==========================
    // HC-SR04
    // ==========================

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    // ==========================
    // MPU6050
    // ==========================

    Wire.begin();

    mpu.initialize();

    // ==========================
    // GPS
    // ==========================

    gpsSerial.begin(9600);

    Serial.println("SMART BLACKBOX SENSOR HUB STARTED");
}



// ==========================================
// READ ULTRASONIC
// ==========================================

void readUltrasonic()
{
    digitalWrite(TRIG_PIN, LOW);

    delayMicroseconds(2);

    digitalWrite(TRIG_PIN, HIGH);

    delayMicroseconds(10);

    digitalWrite(TRIG_PIN, LOW);

    long duration = pulseIn(ECHO_PIN, HIGH);

    distance_cm = duration * 0.034 / 2.0;
}



// ==========================================
// READ MPU6050
// ==========================================

void readMPU()
{
    mpu.getMotion6(
        &ax,
        &ay,
        &az,
        &gx,
        &gy,
        &gz
    );
}



// ==========================================
// READ GPS
// ==========================================

void readGPS()
{
    while (gpsSerial.available())
    {
        gps.encode(gpsSerial.read());

        if (gps.location.isUpdated())
        {
            latitude = gps.location.lat();

            longitude = gps.location.lng();
        }
    }
}



// ==========================================
// SEND DATA
// ==========================================

void sendData()
{
    Serial.print("DIST:");

    Serial.print(distance_cm);

    Serial.print(",");



    Serial.print("IMU:");

    Serial.print(ax);
    Serial.print(",");

    Serial.print(ay);
    Serial.print(",");

    Serial.print(az);
    Serial.print(",");

    Serial.print(gx);
    Serial.print(",");

    Serial.print(gy);
    Serial.print(",");

    Serial.print(gz);

    Serial.print(",");



    Serial.print("GPS:");

    Serial.print(latitude, 6);

    Serial.print(",");

    Serial.print(longitude, 6);

    Serial.println();
}



// ==========================================
// LOOP
// ==========================================

void loop()
{
    // ==========================
    // READ ALL SENSORS
    // ==========================

    readUltrasonic();

    readMPU();

    readGPS();

    // ==========================
    // SEND TO JETSON
    // ==========================

    sendData();

    delay(100);
}