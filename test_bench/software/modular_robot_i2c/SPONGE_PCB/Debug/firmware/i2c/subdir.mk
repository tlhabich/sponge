################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../firmware/i2c/i2c_slave.c 

OBJS += \
./firmware/i2c/i2c_slave.o 

C_DEPS += \
./firmware/i2c/i2c_slave.d 


# Each subdirectory must supply rules for building sources it contributes
firmware/i2c/%.o firmware/i2c/%.su firmware/i2c/%.cyclo: ../firmware/i2c/%.c firmware/i2c/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I"../../../firmware" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-firmware-2f-i2c

clean-firmware-2f-i2c:
	-$(RM) ./firmware/i2c/i2c_slave.cyclo ./firmware/i2c/i2c_slave.d ./firmware/i2c/i2c_slave.o ./firmware/i2c/i2c_slave.su

.PHONY: clean-firmware-2f-i2c

