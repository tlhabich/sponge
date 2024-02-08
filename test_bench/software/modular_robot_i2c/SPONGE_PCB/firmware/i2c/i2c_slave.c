/*
 * i2c_slave.c
 *
 *  Created on: Jul 4, 2023
 *      Author: habich
 */


#include "i2c_slave.h"

I2C_Slave *active_i2c_slave = NULL;

uint8_t receiveBuffer[128] = {0};
uint8_t sendBuffer[128] = {0};

void nop() {

}

void HAL_I2C_ListenCpltCallback(I2C_HandleTypeDef *hi2c) {
	if (active_i2c_slave != NULL) {
		i2c_listenCompleteCallback(active_i2c_slave);
	}
}

void HAL_I2C_AddrCallback(I2C_HandleTypeDef *hi2c, uint8_t TransferDirection,
		uint16_t AddrMatchCode) {
	if (active_i2c_slave != NULL) {
		i2c_addrCallback(active_i2c_slave, TransferDirection, AddrMatchCode);
	}
}

void HAL_I2C_SlaveRxCpltCallback(I2C_HandleTypeDef *hi2c) {
	if (active_i2c_slave != NULL) {
		i2c_rxCompleteCallback(active_i2c_slave);
	}
}

void HAL_I2C_SlaveTxCpltCallback(I2C_HandleTypeDef *hi2c) {
	if (active_i2c_slave != NULL) {
		i2c_txCompleteCallback(active_i2c_slave);
	}
}

void HAL_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c) {
	if (active_i2c_slave != NULL) {
		i2c_errorCallback(active_i2c_slave);
	}
}

void HAL_I2C_AbortCpltCallback(I2C_HandleTypeDef *hi2c) {
	if (active_i2c_slave != NULL) {
		i2c_abortCompleteCallback(active_i2c_slave);
	}
}


void i2c_slave_init(I2C_Slave* slave) {
	slave->sendBuffer = sendBuffer;
	slave->receiveBuffer = receiveBuffer;


	slave->received_bytes=0;
	slave->sent_bytes=0;
	slave->bufferAddress=0;


	active_i2c_slave = slave;


}

void i2c_slave_start(I2C_Slave* slave) {
	HAL_I2C_EnableListen_IT(slave->hi2c);
}


void i2c_addrCallback(I2C_Slave* slave,uint8_t TransferDirection, uint16_t AddrMatchCode) {
	if (TransferDirection == I2C_DIRECTION_TRANSMIT) {
			slave->direction = I2C_SLAVE_DIRECTION_TRANSMIT;
			if (slave->received_bytes == 0) {
				HAL_I2C_Slave_Seq_Receive_IT(slave->hi2c, &slave->bufferAddress, 1,
				I2C_NEXT_FRAME);
			} else {
				nop();
			}

		} else if (TransferDirection == I2C_DIRECTION_RECEIVE) {
			slave->direction = I2C_SLAVE_DIRECTION_RECEIVE;
			HAL_I2C_Slave_Seq_Transmit_IT(slave->hi2c,
					&slave->receiveBuffer[slave->bufferAddress], 1, I2C_NEXT_FRAME);
		}
}

void i2c_listenCompleteCallback(I2C_Slave* slave){
	uint8_t rxbytes = slave->received_bytes;
	slave->received_bytes = 0;
	slave->sent_bytes = 0;
	HAL_I2C_EnableListen_IT(slave->hi2c);

	if (slave->rxCallback != NULL) {
		slave->rxCallback(slave->receiveBuffer, rxbytes);
	}



}
void i2c_rxCompleteCallback(I2C_Slave* slave){
	slave->received_bytes++;
		if (slave->received_bytes > 1) {

			slave->bufferAddress++;

		}
		HAL_I2C_Slave_Seq_Receive_IT(slave->hi2c,
				&slave->receiveBuffer[slave->bufferAddress], 1, I2C_NEXT_FRAME);
}
void i2c_txCompleteCallback(I2C_Slave* slave){
	slave->bufferAddress++;
	slave->sent_bytes++;
		HAL_I2C_Slave_Seq_Transmit_IT(slave->hi2c,
				&slave->sendBuffer[slave->bufferAddress], 1, I2C_NEXT_FRAME);

}
void i2c_errorCallback(I2C_Slave* slave){
	nop();
	HAL_I2C_EnableListen_IT(slave->hi2c);
}
void i2c_abortCompleteCallback(I2C_Slave* slave){

}
