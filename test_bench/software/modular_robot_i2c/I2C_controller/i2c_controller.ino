#include <Arduino.h>
#include <Wire.h>
#include "EasyCAT.h"

#include "elapsedMillis.h"

EasyCAT EASYCAT; //create EasyCAT object

//int addr_arr [] = {0x1}; //I2C adresses. Configure before running

//-------------------------------------------
//Uncomment only 1 of the following Lines !!!
//int addr_arr [] = {0x1}; //1 Actuator
//int addr_arr [] = {0x1,0x2}; //2 Actuators
int addr_arr [] = {0x1,0x2,0x3}; //3 Actuators
//-------------------------------------------

unsigned long lastMicros = 0;
unsigned char EcatState;

elapsedMicros timer;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); 
  delay(1000);
  Serial.println("\nMaster started");
  delay(10);
  
  Wire.begin();               //Initialize I2C
  Wire.setClock(400000);      // Set clock speed: default 100kHz; 400kHz; 700kHz

  
  if(EASYCAT.Init()==true){   //Initialize EasyCAT object
    Serial.println("EasyCAT initialization complete");
    delay(100);
  }
  else{
    Serial.println("EasyCAT initialization failed");
    delay(100);
  } 

}

void loop() {
/* //Debug lines: sends specified numbers to target so that reaction can be seen
uint8_t buffer[5] = {1,0,3,4,5};
uint8_t readBuffer[15] = {0};
Wire.beginTransmission(0x01);
Wire.write(0);
Wire.write(0);
Wire.endTransmission();
Wire.beginTransmission(0x01);
Wire.write(0);
Wire.endTransmission();
Wire.requestFrom(0x01, 2);

for (uint8_t a = 0;a<2;a++){
  readBuffer[a] = Wire.read(); 
  Serial.println(readBuffer[a]);
}
delay(1000);
*/

if (timer >= 1000) {
  timer = 0;
  constexpr int i2cAddrLen = sizeof(addr_arr)/sizeof(addr_arr[0]);
   for (size_t i=0; i<i2cAddrLen; i++){ //Loop over all targets
    //send valve commands

    Wire.beginTransmission(addr_arr[i]); // select target
    Wire.write(0); //send to buffer address 0 of target (don't change). 0 is buffer start
    for (uint8_t a = 0; a<2; a++){
      Wire.write(EASYCAT.BufferOut.Byte[i*2 + a]);
    } //send command: 2 Byte per Slave
    Wire.endTransmission(); //stop writing

  //reading sensor signal
    // tell target, which buffer address to read from
    Wire.beginTransmission(addr_arr[i]); // select target
    Wire.write(0); // read from buffer address 0 of target (don't change). 0 is buffer start
    Wire.endTransmission();

    Wire.requestFrom(addr_arr[i],2); //request sensor signals. 12 Bit ADC therefore 2 Bytes per target  
    for (uint8_t b = 0; b<2; b++){
      EASYCAT.BufferIn.Byte[i*2 + b] = Wire.read(); //buffer for sending via EtherCAT
   }
   }
  //delay(1000);
  EcatState = EASYCAT.MainTask(); //Send end receive via EtherCAT

}

}
