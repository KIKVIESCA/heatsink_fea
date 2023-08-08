# HEATSINK FINITE ELEMENT ANALYSIS
Thermal finite element analysis using python libraries.

## Dimensional specification

Heatsink dimensions have to be specified for simulation:
* The baseplate width is obtained when selecting a heatsink.
* The baseplate height must be specified for vertical convection. Z axis increase from bottom to top, in the direction of airflow.
* The baseplate thickness must be specified for thermal conduction.

![Heatsink dimensions](./img/baseplate-dimensions-mbf.png)

## Mehsgrid

The heatsink is subsequently divided in rectangles. The simulation ends when a loop takes too much time.

![Map partititon](./img/meshgrid.png)

## Meshgrid variables

Each element is assigned a temperature. The starting temperature used is the ambient temperature increased by the temperature increase specified in the heatsink datasheet. On resizing, the temperature of each element stays. On loop, temperature differente is calculated trough admittance megamatrix.

The power generation matrix is calculated for element. Intersection of heat sources with each element is multiplied by the heat density.

Both the radiation and the convection admittance are calculated for each element base on the vertical position. Emissivity of 0.9 is used for the radiation.

Conduction for adjacent elements is calculated using the contact length * baseplate thickness / distance between centers of elements.

### Natural convection grid

The datasheet resistance (Rth(z) for z in range(0, z_max)) is inverted for admittance values (Yth(z) for z in range(0, z_max)). Element admittance is the local increase of the admittance. Total admittance must be equal to the sum of the admittance of all elements.

![Linear admittance](./img/linear_admittance.png)

However, this does not account for distributed heat.
Therefore, instead of getting the element adimttance by scaling z values (interpolate (z, z_datasheet, dy_datasheet)), it can be obtained by scaling the sum of the temperatures under z (sum(t under z), sum(t under z_datasheet), dy_datasheet).

![Datasheet comparison](./img/datasheet_comparison.png)

## Thermal loops

Temperatures, convection, radiation, and power generation matrices are flattened. The resulting vector will start with the element on the bottom left corner, and will end with the element on the top right corner of the heatsink.
The element admittance megamatrix is created with the flattened convection+radiation admittance in the diagonal.
The conduction matrix is added, considering all permutations of adjacent conduction within the boundary limits.
Finally, the temperature is calculated by inverting the megamatrix and multiplying it by the flattened power generation.

## Report generation

Thermal palettes are used to map the temperature results into an image. The minimum temperature of the map is the ambient temperature to allow easy comparison with thermal images. The maximum temperature of the map is the maximum temperature simulated.