/*
 * i2c_slave.h
 *
 *  Created on: Jul 4, 2023
 *      Author: habich
 */

#ifndef I2C_SLAVE_H_
#define I2C_SLAVE_H_

#include "stm32f4xx_hal.h"

typedef enum I2C_Slave_direction {
	I2C_SLAVE_DIRECTION_TRANSMIT, I2C_SLAVE_DIRECTION_RECEIVE
}I2C_Slave_direction;



extern uint8_t receiveBuffer[128];
extern uint8_t sendBuffer[128];

typedef struct I2C_Slave {
	I2C_HandleTypeDef *hi2c;
	I2C_Slave_direction direction;
	uint8_t received_bytes;
	uint8_t sent_bytes;
	uint8_t *receiveBuffer;
	uint8_t *sendBuffer;
	uint8_t bufferAddress;
	uint8_t address;
	void (*rxCallback) (uint8_t*, uint8_t);
} I2C_Slave;




extern I2C_Slave *active_i2c_slave;

void i2c_slave_init(I2C_Slave* slave);
void i2c_slave_start(I2C_Slave* slave);


void i2c_addrCallback(I2C_Slave* slave,uint8_t TransferDirection, uint16_t AddrMatchCode);
void i2c_listenCompleteCallback(I2C_Slave* slave);
void i2c_rxCompleteCallback(I2C_Slave* slave);
void i2c_txCompleteCallback(I2C_Slave* slave);
void i2c_errorCallback(I2C_Slave* slave);
void i2c_abortCompleteCallback(I2C_Slave* slave);

#endif /* I2C_SLAVE_H_ */
