# Interactively model the kitchen of the world’s largest oil field !

The model uses a Neural Network trained for the Jafurah basin, which is the kitchen for the southern fetch area of Ghawar, the world’s largest oil field, and which is also an area of current Shale Gas development. The results of the wider model will be shown by Andrew Pepper of This is Petroleum Systems (t!Ps) at the AAPG GeoScience Technology Workshop [Source Rocks of the Middle East: A World Class Resource for Unconventional?](https://www.aapg.org/global/middleeast/events/virtual/articleid/55566/source-rocks-of-the-middle-east) taking place from 26-28 September 2022 in Manama, Bahrain.

The Neural Network has been trained following the methodology initially described in
[Petroleum Systems Through Economic Assessment of the Permian Basin, Texas Aided By a Cognitive System for Geosciences](https://www.searchanddiscovery.com/abstracts/html/2018/ice2018/abstracts/3005524.html), Laigle et al., 2018. It delivers instant temperature, thermal stress and maturity predictions at the ~ 99.9% level of accuracy compared to a full physics basin simulator.

The default depths and stratigraphic scheme in the app were derived by t!Ps, based initially on Well E in
[A basin modeling study of the Jafurah Sub-Basin, Saudi Arabia: Implications for unconventional hydrocarbon potential of the Jurassic Tuwaiq Mountain Formation](https://www.sciencedirect.com/science/article/abs/pii/S0166516216303032?via%3Dihub)
A. Hakami and S. Inan, 2016

Understand the impact of the parameters:

* **Depth uncertainty** : scales the entire geological column by the factor selected; for example reflecting uncertainty in seismic depth conversion
* **Neogene erosion** : varies the amount of late Paleogene/early Neogene erosion to investigate the timimg of maximum thermal stress and maturity 
* **Basement parameters** : vary the crust and upper mantle structure. Crust thickness and Radiogenic Heat Production (RHP) determine the crustal contribution to heat flow while the depth to the aestenosphere changes the distance to the lower boundary temperature of 1330 degrees C

on present temperature, maximum-reading thermal stress and thermal stress indicators Ro (not recommended in marine aquatic organofacies!) and Tmax. The color bar represents the typical instantaneous maturity intervals corresponding to expulsion of low GOR early oil (dark green) through volatile oil (yellow) to dry gas (red) and beyond the gas window (grey).

You can also load your own model as an Excel file for instant, full physics, temperature and maturity results.

***Have fun! - from the t!Ps and Daisi Teams***

For any questions / comments on this app, please reach out to Andrew Pepper (asp.tips@me.com)

To call it programatically in Python:

```python
import pydaisi as pyd
import numpy as np
import pandas as pd

data = pd.read_excel('template_daisi_SA.xlsx')['Depth'].values # Depths of the 43 key surfaces
data = np.append(data, 1000) # Neogene erosion amount [m]
data = np.append(data, 25000) # Crust thickness [m](Base sediment to Moho)
data = np.append(data, 1e-7) # Lower crust RHP [W/m3](34% of total crust thickness)
data = np.append(data, 70000) # Upper mantle thickness [m](Moho to LAB)
data = np.append(data, 3e-6) # Upper crust RHP [W/m3](66% of total crust thickness)
data = np.append(data, 24) # Present day surface temperature

data = data.reshape((49, 1, 1)) # 1D geological column with 49 parameters. 
# Note that you can also pass a grids based model of size (nx,ny). In that case, data.shape should be (49, ny, nx)

neural_network_jarufah_basin = pyd.Daisi("Neural Network Jafurah Basin")
result = neural_network_jarufah_basin.get_predictions(data, variable = 'temperature').value # variable = 'temperature' or 'maturity'. 
# Values are returned at cells mid points.
```
