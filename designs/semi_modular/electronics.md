---
title: Electronics (optional)
parent: Semi-Modular SPONGE
grand_parent: Designs
nav_order: 2
---

# Electronics (optional)
The optional PCB (mounted opposite each encoder in the joint) handles the modular communication of the encoders with the [test bench](https://tlhabich.github.io/sponge/test_bench/) via I2C. It comprises an AD-Converter (ADC) ADS122C04 to convert the analogue output of each encoder into a digital signal. Each PCB is configured as an I2C target (slave). As the I2C controller (master) serves a NUCLEO-F401RE development board, that is located outside the robot. Attached to it is a Bausano EasyCAT-shield that enables a connection from the NUCLEO board to the EtherCAT bus for communication with the test bench.

<p align="center">
  <img src="images/../../../images/PCB_semi_modular.png" width=600>
</p>

## Circuit Board

<img src="images/../../../images/circuit_diagram_semi_modular.png">

 The [circuit diagram](https://github.com/tlhabich/sponge/blob/master/images/circuit_diagram_semi_modular.png) shows the structure of the board. The ADC is composed of 16 pins. The device is powered with 3.3 V, and requires capacitors to ensure stable operation. The I2C lines (SDA and SCL) are provided with pull-up resistors. Given that only a single pull-up resistor is necessary for each line, jumper pins were incorporated into each circuit. Pull-up resistors can be integrated into the circuit by soldering them directly to it. The jumper pins can then be used for necessary connections. The ADC can be configured to one of sixteen possible I2C addresses via pins A0 and A1 [(Table 14 in datasheet)](https://www.ti.com/lit/ds/symlink/ads122c04.pdf?ts=1755394140893&ref_url=https%253A%252F%252Fwww.ti.com%252Fproduct%252FADS122C04). This was also realized with jumper pins, thereby enabling flexible I2C-address assignment.

The Hall encoder is connected to the interface using a 3x1 Molex PicoBlade connector. Voltage dividers with resistances of 200kOhm and 100kOhm are necessary for downscaling the encoder signal (from 10 V to 3.3 V). It should be noted that the use of a buffer is unnecessary, as the ADC is already equipped with an integrated buffer. 5x1 Molex PicoBlade connectors facilitates the (daisy chain) connection of the PCBs. The PCB of the first soft actuator is connected with the NUCLEO development board.

## Components
### On PCB

| name  | type  | quantity <br> per PCB|
|:----:   |:----:   |:----:   |
| [GCD21BR72A104KA01L](https://www.mouser.de/ProductDetail/Murata-Electronics/GCD21BR72A104KA01L?qs=QzBtWTOodeUgjSVpOGt6MA%3D%3D) | capacitor 0.1μF  | 2 (C1, C2)                |
| [TNPW0805200KBHEA](https://www.mouser.de/ProductDetail/Vishay/TNPW0805200KBHEA?qs=vmHwEFxEFR89EiIUS0B7jA%3D%3D)                 | resistor 200kOhm | 1 (R1)                |
| [ERA6AEB104V](https://www.mouser.de/ProductDetail/Panasonic/ERA-6AEB104V?qs=1VWA5LkbEapLJhdoEEiQ2A%3D%3D)                       | resistor 100kOhm | 1 (R2)                |
| [TNPW08051K50BEEA](https://www.mouser.de/ProductDetail/Vishay/TNPW08051K50BEEA?qs=Db%252BqmG6P6GaxwcmU1luVmQ%3D%3D)             | resistor 1.5kOhm | 2 (RPSCL, RPSDA)               |
| [ADS122C04IPW](https://www.mouser.de/ProductDetail/Texas-Instruments/ADS122C04IPW?qs=7EBvPakHacVHFL5PAbuAZg%3D%3D)              | AD converter     | 1                |
| [TSW-104-07-F-D](https://www.mouser.de/ProductDetail/Samtec/TSW-104-07-F-D?qs=rHlcMk0NooJDzMj2iWZbbA%3D%3D)                     | jumpers 4x2      | 2                |
| [61300211121](https://www.mouser.de/ProductDetail/Wurth-Elektronik/61300211121?qs=t4813l51qx%252B1A5GYDQxPlw%3D%3D)             | jumpers 1x2      | 2                |
| [PRT-09044](https://www.mouser.de/ProductDetail/SparkFun/PRT-09044?qs=sGAEpiMZZMtyU1cDF2RqUE5bpLI3XpN5X9BKNZuaFo4%3D)           | jumper pin       | 2*               |
| [530470510](https://www.mouser.de/ProductDetail/Molex/53047-0510?qs=KC2ywxza1koXSHAr3dqR%2FQ%3D%3D)                             | Molex 5x1 FE   | 2                |
| [530470310](https://www.mouser.de/ProductDetail/Molex/53047-0310?qs=WvGHgJyR8NxkapfSzLYEEg%3D%3D)                               | Molex 3x1 FE   | 1                |

\*4 for the **first PCB** (pull-up resistor connections)
### Other

|name | type |quantity |
|:----|:----:|:----:|
|[NUCLEO-F401RE](https://www.mouser.de/ProductDetail/STMicroelectronics/NUCLEO-F401RE?qs=sGAEpiMZZMuqBwn8WqcFUv%2FX0DKhApUpi46qP7WpjrffIid8Wo1rTg%3D%3D)| development board | 1|
|[EasyCAT shield](https://www.bausano.net/shop/en/home/16-arduino-ethercat.html)| EtherCAT interface| 1|
|-| UART-TTL USB adapter <br> (optional)|1|
| [15134-0501](https://www.mouser.de/ProductDetail/Molex/15134-0501?qs=lQAVKuKFhkITxQQuK1BAAQ%3D%3D)| Molex 5x1 cable M-M                | 1        |
| [15134-0301](https://www.mouser.de/ProductDetail/Molex/15134-0301?qs=lQAVKuKFhkL4Aonld%2FWL2g%3D%3D)                                                    | Molex 3x1 Cable M-*                | 0.5      |
| -| jumper cable M-**                    | 5        |

\*Only the M-side is required. Cable is cut in half and soldered to the encoder cable.

\**Only the M-sides are required. Cables are cut in half and soldered to the Molex 5x1 cable M-M.

## Manufacturing
The [Gerber Files](/sponge/downloads/SPONGE_SemiModular_PCB_Gerber.zip) can be directly sent to a PCB manufacturer. After that, we soldered the components onto the board ourselves.
## Software
Please refer to the [software section of the modular PCB](https://tlhabich.github.io/sponge/designs/modular/electronics.html). All necessary files for extending the [existing test-bench software](https://github.com/tlhabich/sponge/tree/main/test_bench/software) can be found in this [folder](https://github.com/tlhabich/sponge/tree/main/test_bench/software/semimodular_robot_i2c). **TODO: Short Description**

According to its datasheet, the AD converter has an offset that can be calibrated by changing its configuration. In order to enable/disable this functionality, change the following code line:

```c
// Enable offset calibration?
static const bool ENABLE_CALIBRATION = false;
```

The number of samples to be used for the calibration could also be set:
```c
// Number of calibration samples
const int CAL_SAMPLES = 20;
```

The registries of the AD converter must be configured before use. This is done in the setup function of the program:
- Reg 0: (`0x81`)
	- AIN0 vs AVSS
	- gain=1
	- PGA bypass
- Reg 1: (`0xDC`)
	- 2000 SPS
	- continuous mode

Reg 2 and Reg 3 are left at default. The I2C speed mode is set to "Fast Mode Plus" (1 Mbit/s).
