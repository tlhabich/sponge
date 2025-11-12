#include <Arduino.h>
#include <Wire.h>
#include "EasyCAT.h"
#include "elapsedMillis.h"

// =============================================================================
// DEFINE CONSTANTS
// =============================================================================

// Enable Offset calibration?
static const bool ENABLE_CALIBRATION = false;

// ADS commands
// Single-register write commands
static const uint8_t WREG_REG0 = 0x40; 
static const uint8_t WREG_REG1 = 0x44; 
// Single-register read commands
static const uint8_t RREG_REG0 = 0x20;
static const uint8_t RREG_REG1 = 0x24;
// Other commands
static const uint8_t CMD_START  = 0x08;  
static const uint8_t CMD_RDATA  = 0x10;  
static const uint8_t CMD_RESET  = 0x06;  

//Configurations
// Normal configuration for Register0 & Register1
static const uint8_t ADS_CFG0_VAL = 0x81; // AIN0 vs AVSS, gain=1, PGA bypass
static const uint8_t ADS_CFG1_VAL = 0xDC; // 2000 SPS, continuous mode
// Calibration commands: for offset calibration, to short inputs internally to (AVDD+AVSS)/2, with gain=1, PGA bypassed, MUX=1110 => bottom nibble = 0xE
static const uint8_t ADS_CFG0_OFFSET_CAL = 0xE0; 
// Number of calibration samples
const int CAL_SAMPLES = 20;

// I2C Speed
static const int I2C_SPEED = 1000000; // 1 MHz

// =============================================================================
// EASYCAT
// =============================================================================
EasyCAT EASYCAT; // Create EasyCAT object
unsigned char EcatState;

// Timer to schedule reading (in microseconds)
elapsedMicros timer;

// =============================================================================
// I2C ADDRESS LIST
// =============================================================================
static const uint8_t possibleAdsAddresses[] = {
  0x40, 0x41, 0x42, 0x43,
  0x44, 0x45, 0x46, 0x47,  
  0x48, 0x49, 0x4A, 0x4B,
  0x4C, 0x4D, 0x4E, 0x4F
};
static bool adsPresent[sizeof(possibleAdsAddresses)/sizeof(possibleAdsAddresses[0])];
static size_t adsCount = 0;

// For storing one offset value per ADS
static int32_t adsOffset[sizeof(possibleAdsAddresses)/sizeof(possibleAdsAddresses[0])] = {0};

// =============================================================================
// FUNCTIONS
// =============================================================================

// Write to ADS Register
void ads_writeSingleRegister(uint8_t adsAddr, uint8_t regCommand, uint8_t value)
{
  Wire.beginTransmission(adsAddr);
  Wire.write(regCommand);
  Wire.write(value);
  Wire.endTransmission();
}

// Read from ADS Register
uint8_t ads_readSingleRegister(uint8_t adsAddr, uint8_t regCommand)
{
  Wire.beginTransmission(adsAddr);
  Wire.write(regCommand); 
  Wire.endTransmission(false);

  Wire.requestFrom((int)adsAddr, 1);
  while (!Wire.available()) {}
  return Wire.read();
}

// Send command to ADS
void ads_sendCommand(uint8_t adsAddr, uint8_t cmd)
{
  Wire.beginTransmission(adsAddr);
  Wire.write(cmd);
  Wire.endTransmission();
}

// Read data from ADS
int32_t ads_readData(uint8_t adsAddr)
{
  ads_sendCommand(adsAddr, CMD_RDATA);

  // Read 3 bytes
  Wire.requestFrom((int)adsAddr, 3);
  while (Wire.available() < 3) {}
  uint8_t msb = Wire.read();
  uint8_t mid = Wire.read();
  uint8_t lsb = Wire.read();

  // Combine into a 24-bit raw
  int32_t raw = ((int32_t)msb << 16) | ((int32_t)mid << 8) | (int32_t)lsb;
  return raw;
}

// Configure a single ADS
void ads_initDevice(uint8_t adsAddr)
{
  // Reset
  ads_sendCommand(adsAddr, CMD_RESET);
  delay(5);

  // Write config register 0 & 1
  ads_writeSingleRegister(adsAddr, WREG_REG0, ADS_CFG0_VAL);
  ads_writeSingleRegister(adsAddr, WREG_REG1, ADS_CFG1_VAL);

  // Read back
  uint8_t c0 = ads_readSingleRegister(adsAddr, RREG_REG0);
  uint8_t c1 = ads_readSingleRegister(adsAddr, RREG_REG1);
  Serial.print("ADS 0x");
  Serial.print(adsAddr, HEX);
  Serial.print(": CFG0=0x");
  Serial.print(c0, HEX);
  Serial.print(", CFG1=0x");
  Serial.println(c1, HEX);

  // Start continuous conversions
  ads_sendCommand(adsAddr, CMD_START);
  delay(5);
}

// Determine Offset
int32_t ads_calibrateOffset(uint8_t adsAddr, int nSamples)
{
  // Read the current register0 setting
  uint8_t oldReg0 = ads_readSingleRegister(adsAddr, RREG_REG0);

  // Write the new MUX setting: 0xE0
  ads_writeSingleRegister(adsAddr, WREG_REG0, ADS_CFG0_OFFSET_CAL);
  delay(5);

  // Take multiple readings and accumulate
  int64_t sum = 0;  // Using 64-bit to avoid potential overflow with multiple samples
  for(int i = 0; i < nSamples; i++)
  {
    int32_t rawVal = ads_readData(adsAddr);
    sum += rawVal;
    delay(2); // small delay between samples, just in case
  }
  int32_t offset = (int32_t)(sum / nSamples);

  // Restore original register 0
  ads_writeSingleRegister(adsAddr, WREG_REG0, oldReg0);
  delay(5); // small delay to guarantee correct writing

  return offset;
}

// =============================================================================
// SETUP & LOOP
// =============================================================================
void setup() 
{
  Serial.begin(115200);
  delay(1000);
  Serial.println("Master started");

  // Initialize I2C
  Wire.begin();
  Wire.setClock(I2C_SPEED);
  Serial.print("I²C Speed: ");
  Serial.println(I2C_SPEED);

  // Initialize EasyCAT object
  if(EASYCAT.Init()==true){  
    Serial.println("EasyCAT initialization complete");
    delay(100);
  }
  else{
    Serial.println("EasyCAT initialization failed");
    delay(100);
  } 

  // Scan for ADS devices
  size_t nPossible = sizeof(possibleAdsAddresses)/sizeof(possibleAdsAddresses[0]);
  for (size_t i = 0; i < nPossible; i++) {
    uint8_t addr = possibleAdsAddresses[i];
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      // Found an ADS device
      adsPresent[i] = true;
      adsCount++;
      Serial.print("Found ADS at Address: 0x");
      Serial.print(addr, HEX);
      Serial.print(" ------- ");

      // Initialize the device
      ads_initDevice(addr);

      // Perform offset calibration once at startup, if calibration is enabled
      if (ENABLE_CALIBRATION == true) {
        int32_t offsetVal = ads_calibrateOffset(addr, CAL_SAMPLES);
        adsOffset[i] = offsetVal;
        Serial.print("Calibrated offset (ADS 0x");
        Serial.print(addr, HEX);
        Serial.print("): ");
        Serial.println(offsetVal, HEX);
      }
      
    } 
    else {
      adsPresent[i] = false;
    }
  }
  Serial.print("ADS122C04 devices found: ");
  Serial.println(adsCount);
}

void loop() 
{
  // Periodically read each ADS
  if (timer >= 1000){
    timer = 0;

    size_t nPossible = sizeof(possibleAdsAddresses)/sizeof(possibleAdsAddresses[0]);

    for (size_t i = 0; i < nPossible; i++) {
      if (adsPresent[i]) {
        uint8_t addr = possibleAdsAddresses[i];
        int32_t rawVal = ads_readData(addr);
        
        // Subtract the offset we found at startup, if calubration is enabled
        int32_t returnVal = 0;
        if (ENABLE_CALIBRATION == true){
          returnVal = rawVal - adsOffset[i];
        } else {
          returnVal = rawVal;
        }       

        // Break correctedVal into 3 bytes
        uint8_t b2 = (uint8_t)((returnVal >> 16) & 0xFF);
        uint8_t b1 = (uint8_t)((returnVal >>  8) & 0xFF);
        uint8_t b0 = (uint8_t)( returnVal        & 0xFF);

        // Store in EasyCAT input buffer
        // 3 bytes per device -> indices: 3*i, 3*i+1, 3*i+2
        EASYCAT.BufferIn.Byte[3*i + 0] = b2;
        EASYCAT.BufferIn.Byte[3*i + 1] = b1;
        EASYCAT.BufferIn.Byte[3*i + 2] = b0;
      }
    }

    // EtherCAT communication handler
    EcatState = EASYCAT.MainTask();
  }
}