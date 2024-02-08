#include <Arduino.h>
#include <Wire.h>
#include "EasyCAT.h"

EasyCAT EASYCAT; //create EasyCAT object

int addr_arr [] = {0x1,0x2,0x3}; //I2C adresses. Configure before running

unsigned long lastMicros = 0;
unsigned char EcatState;

TwoWire Wire2(PB9,PB8);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); 
  delay(1000);
  Serial.println("\nMaster started");
  delay(10);
  
  //pinMode(15, OUTPUT);

 
  Wire2.begin();               //Initialize I2C
  Wire2.setClock(400000);      // Set clock speed: default 100kHz; 400kHz; 700kHz

  /*
  if(EASYCAT.Init()==true){   //Initialize EasyCAT object
    Serial.println("EasyCAT initialization complete");
  }
  else{
    Serial.println("EasyCAT initialization failed");
  } 
  */
  
}

void loop() {
//Serial.println("Send...");
//digitalWrite(15, !digitalRead(15));

uint8_t buffer[5] = {1,2,3,4,5};
Wire.beginTransmission(0x81);
for (uint8_t a = 0;a<5;a++){
  Wire.write(buffer[a]);
}

Wire.endTransmission();



delay(100);

  /*
  // put your main code here, to run repeatedly:
  if(lastMicros + 1000 < micros()){ //Timeout condition 1s
    constexpr int i2cAddrLen = sizeof(addr_arr)/sizeof(addr_arr[0]);
    for (size_t i=0; i<i2cAddrLen; i++){ //Loop over all slaves
      Wire.beginTransmission(addr_arr[i]); // send valve commands
        if(i<4){            //for more than 4 slaves Valve commands take up more than 1 Byte. (2 Bit per slave)
          Wire.write(EASYCAT.BufferOut.Byte[0]);
        }
        else {            // Suitable for up to 8 slaves. Extend if more are to be used.
          Wire.write(EASYCAT.BufferOut.Byte[1]);
        }
      Wire.endTransmission();
      int ByteCount = 0;
      Wire.requestFrom(addr_arr[i],2); //Read sensor signals. 12 Bit ADC therefore 2 Bytes per Sensor signal
      while (ByteCount < 2){
        if (Wire.available()){
          EASYCAT.BufferIn.Byte[i*2 + ByteCount] = Wire.read(); //buffer for sending via EtherCAT
          ByteCount++;
        }
        else {
          Serial.println("I2C Transmission failed. Re-request package");
          Wire.requestFrom(addr_arr[i],2);
        }
      }
      lastMicros = micros(); 
    }

  }

  EcatState = EASYCAT.MainTask(); //Send end receive via EtherCAT
*/
}
