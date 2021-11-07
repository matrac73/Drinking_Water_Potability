  _____       _       _    _                             _            
 |  __ \     (_)     | |  (_)                           | |           
 | |  | |_ __ _ _ __ | | ___ _ __   __ _  __      ____ _| |_ ___ _ __ 
 | |  | | '__| | '_ \| |/ / | '_ \ / _` | \ \ /\ / / _` | __/ _ \ '__|
 | |__| | |  | | | | |   <| | | | | (_| |  \ V  V / (_| | ||  __/ |   
 |_____/|_|  |_|_| |_|_|\_\_|_| |_|\__, |   \_/\_/ \__,_|\__\___|_|   
                                    __/ |                             
             _____      _        _ |___/ _ _ _                        
            |  __ \    | |      | |   (_) (_) |                       
            | |__) |__ | |_ __ _| |__  _| |_| |_ _   _                
            |  ___/ _ \| __/ _` | '_ \| | | | __| | | |               
            | |  | (_) | || (_| | |_) | | | | |_| |_| |               
            |_|   \___/ \__\__,_|_.__/|_|_|_|\__|\__, |               
                                                  __/ |               
                                                 |___/               

# Setup du projet

1 . Télécharger **anaconda** et ajouter **conda** au PATH (si ce n'est pas déja fait)

2 . Créer un environement virtuel anaconda avec la bonne version de python :
``` bash
conda create -n drinking_water_potability python=3.8.5
```
3 . Activer l'environement virtuel :
``` bash
conda activate drinking_water_potability
```
4 . Se placer dans le répertoire dans lequel on veut insaller le dépot git :
``` bash
cd ./path/to/repository/
```
5 . Cloner le dépot git :
``` bash
git clone https://github.com/matrac73/Drinking_Water_Potability.git
```
6 . Se placer dans le répertoire qui viens d'être crée :
``` bash
cd ./Drinking_Water_Potability/
```
7 . Télécharger les dépendances :
``` bash
python -m pip install -r requirements.txt
```
8 . Ajouter l'environement à jupyter lab
``` bash
python -m ipykernel install --user --name=drinking_water_potability
```
9 . Lancer Jupyter Lab dans une nouvelle fenêtre de commande anaconda powershell
``` bash
jupyter lab
```
10 . Choisir le bon kernel (en haut à droite) et exectuer les cellules de **drinking_water_potability.ipynb** dans l'ordre


# Information sur le projet

A rendre pour le 1er novembre
Un rapport écrit (5 pages)
Code source du projet (sur GitHub ou Gitlab)
Une soutenance orale aura lieu en présentiel le jour 12 novembre.

Lien vers descriptif du projet :

https://centralesupelec.edunao.com/pluginfile.php/171459/course/section/30034/Zennaro%20Maxime%20-%20Potabilite%CC%81%20de%20leau.pdf

Lien vers challenge Kaggle :

https://www.kaggle.com/artimule/drinking-water-probability

# Drinking_Water_Potability
Access to safe drinking water is essential to health, a basic human right, and a component of effective policy for health protection. This is important as a health and development issue at a national, regional, and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.

# Content
The drinkingwaterpotability.csv file contains water quality metrics for 3276 different water bodies.

*pH value:*

PH is an important parameter in evaluating the acid-base balance of water. It is also the indicator of the acidic or alkaline condition of water status. WHO has recommended the maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.

*Hardness:*

Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness-producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.

*Solids (Total dissolved solids - TDS):*

Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates, etc. These minerals produced an unwanted taste and diluted color in the appearance of water. This is the important parameter for the use of water. The water with a high TDS value indicates that water is highly mineralized. The desirable limit for TDS is 500 mg/l and the maximum limit is 1000 mg/l which is prescribed for drinking purposes.

*Chloramines:*

Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

*Sulfate:*

Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.

*Conductivity:*

Pure water is not a good conductor of electric current rather’s a good insulator. An increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceed 400 μS/cm.

*Organic_carbon:*

Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

*Trihalomethanes:*

THMs are chemicals that may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm are considered safe in drinking water.

*Turbidity:*

The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of the light-emitting properties of water and the test is used to indicate the quality of waste discharge with respect to the colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

*Potability:*

Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.
