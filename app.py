import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy
import concurrent.futures
import streamlit as st

from io import BytesIO
import pandas as pd
import pydaisi as pyd
# import xlsxwriter

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.patches import Polygon
_lock = RendererAgg.lock

@st.cache
def get_daisi():
	nn = pyd.Daisi("laiglejm/Neural Network Jafurah Basin")
	colormap = pyd.Daisi("laiglejm/Interactive basin modeling Jafurah Basin")
	return nn, colormap



lim_oil_of = {'A':0.0625, 'B':0.125, 'C':0.1875, 'DE':0.28125, 'F':0.375}
lim_oil_sts = {'A':100, 'B':110, 'C':120, 'DE':135, 'F':150}

def get_cmp(OF = 'A'):

	bnd = lim_oil_of[OF]
	
	if OF != 'F':
		cdict = {'red': [(0.0, 0.0078, 0.0078),
						(bnd, 0.0078, 0.0078),
						(bnd, 0.0, 0.0),
						(0.34375, 1.0, 1.0),
						(0.46875, 1.0, 1.0),
						(0.8125, 1.0, 1.0),
						(0.81251, 0.85, 0.85),
						(1.0, 0.85, 0.85)],
				'green': [(0.0, 0.0078, 0.0078),
						(bnd, 0.0078, 0.0078),
						(bnd, 0.58, 0.58),
						(0.34375, 1.0, 1.0),
						(0.46875, 0.0, 0.0),
						(0.8125, 0.0, 0.0),
						(0.81251, 0.85, 0.85),
						(1.0, 0.85, 0.85)],
				'blue': [(0.0, 1.0, 1.0),
						(bnd, 1.0, 1.0),
						(bnd, 0.0, 0.0),
						(0.34375, 0.0, 0.0),
						(0.46875, 0.0, 0.0),
						(0.8125, 0.0, 0.0),
						(0.81251, 0.85, 0.85),
						(1.0, 0.85, 0.85)]
				}
	else:
		cdict = {'red': [(0.0, 0.0078, 0.0078),
						(bnd, 0.0078, 0.0078),
						(bnd, 0.0, 0.0),
						(bnd, 1.0, 1.0),
						(0.46875, 1.0, 1.0),
						(0.8125, 1.0, 1.0),
						(0.81251, 0.85, 0.85),
						(1.0, 0.85, 0.85)],
				'green': [(0.0, 0.0078, 0.0078),
						(bnd, 0.0078, 0.0078),
						(bnd, 0.58, 0.58),
						(bnd, 1.0, 1.0),
						(0.46875, 0.0, 0.0),
						(0.8125, 0.0, 0.0),
						(0.81251, 0.85, 0.85),
						(1.0, 0.85, 0.85)],
				'blue': [(0.0, 1.0, 1.0),
						(bnd, 1.0, 1.0),
						(bnd, 0.0, 0.0),
						(bnd, 0.0, 0.0),
						(0.46875, 0.0, 0.0),
						(0.8125, 0.0, 0.0),
						(0.81251, 0.85, 0.85),
						(1.0, 0.85, 0.85)]
				}

	cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)

	return cmp

path = 'data/'

well_depth_df = pd.read_excel("data/well_depth.xlsx")
w_depth = well_depth_df['Depth (m)'].values

f = 'data/STS_ezRo.csv'
ro_sts = pd.read_csv(f, sep=';')
tmax_of = pd.read_excel("data/STS_Tmax_correl.xlsx")
tmax_headers = tmax_of.columns.values.tolist()

with open('DAISI.md', 'r') as f:
	title = f.readlines()[0]
with open('DAISI.md', 'r') as f:
	summary_md = f.read()[len(title):]

##################################################################################################

def get_sts(x, y, sts_array, depth):
	return np.interp(y, depth, sts_array)

##################################################################################################

def create_sts_map(depth, sts, sts_min, sts_max):
	y = np.arange(depth[0], depth[-1], 10)
	x = np.arange(sts_min, sts_max, 1)
	X, Y = np.meshgrid(x, y)

	sts_map = get_sts(X, Y, sts, depth)

	return sts_map

##################################################################################################

def predict(geol_column, daisi):
	temperature, maturity = daisi.get_all_predictions(data = geol_column).value

	return temperature, maturity

##################################################################################################
def st_ui():
	st.set_page_config(layout = "wide")

	# neural_network_jarufah_basin = get_daisi()
	neural_network_jarufah_basin, colormap = get_daisi()
	if 'of_select' not in st.session_state:
		st.session_state.of_select = -1

	layers_dict = {0: "Quaternary",
						1: "Tertiary",
						2: "Cretaceous",
						3: "Arab",
						4: "Jubaila",
						5: "Hanifa",
						6: "Tuwaiq",
						7: "Fadhili",
						8: "Marrat & Dhruma",
						9: "Sudair, Jihl & Minjur",
						10: "Khuff",
						11: "Wajid"

						}

	layers_wrap = {0: [0],
					1: [1, 2, 3, 4, 5, 6],
					2: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
					3: [22, 23, 24, 25, 26, 27, 28, 29],
					4: [30, 31],
					5: [32, 33, 34],
					6: [35,36],
					7: [37],
					8: [38],
					9: [39],
					10: [40],
					11: [41]
	}
	indx_to_keep = [0, 1, 7, 22, 30, 32, 35, 37, 38, 39, 40, 41, 42]

	geol_column = [d for d in w_depth]
	st.title(title[2:])

	st.sidebar.image('data/TIPS_Logo_mention.png')

	user_file = st.sidebar.file_uploader("Upload a geological column (Excel format)")
	if user_file is not None:
		user_depths = pd.read_excel(user_file)
		user_depths = user_depths['Depth'].values
		geol_column = [d for d in user_depths]

	template_data = deepcopy(well_depth_df)
	towrite = BytesIO()
	downloaded_file = template_data.to_excel(towrite, encoding='utf-8', index=False, header=True) # write to BytesIO buffer
	st.sidebar.download_button(label="Download the geological column template", data=towrite.getvalue(), file_name='template_daisi_SA.xlsx', mime="application/vnd.ms-excel")

	st.sidebar.write("Expand the 'Summary' section in the main window for more info on the below parameters")

	present_day_temperature = st.sidebar.slider("Present Day SWIT (C)", 10, 40, 24)

	depth_uncertainty = st.sidebar.slider('Depth Uncertainty (%). Default = 0%)', -50,50,0)
	depth_uncertainty /= 100

	erosion = st.sidebar.slider('Neogene erosion (default = 1000m)',0, 2000, 1000)

	for i in range(1, len(geol_column)):
			geol_column[i] *= (1 + depth_uncertainty)
	
	geol_column.append(erosion)

	st.sidebar.latex(r"{\rm Basement\, parameters}")

	crust_dict = dict()
	crust_dict["Free selection"] = {"Crust Thickness" : [10, 40, 30], 
							   "Upper Crust RHP" : [0., 5., 1.5],
							   "Lower Crust RHP" : [0., 2., 0.2],
							   "Mantle Thickness": [50, 120, 80]}
	crust_dict["Oceanic Crust"] = {"Crust Thickness" : [8, 15, 11], 
							   "Upper Crust RHP" : [0., 0.1, 0.],
							   "Lower Crust RHP" : [0., 0.1, 0.],
							   "Mantle Thickness": [50, 70, 60]}
	crust_dict["Transitional Crust"] = {"Crust Thickness" : [10, 20, 15], 
							   "Upper Crust RHP" : [0., 2., 1.5],
							   "Lower Crust RHP" : [0., 0.5, 0.2],
							   "Mantle Thickness": [50, 90, 70]}
	crust_dict["Continental Crust"] = {"Crust Thickness" : [18, 40, 25], 
							   "Upper Crust RHP" : [1., 5., 3.],
							   "Lower Crust RHP" : [0., 2., 1.],
							   "Mantle Thickness": [80, 120, 100]}

	crust_type = "Free selection"
	mmd = crust_dict[crust_type]["Crust Thickness"]
	crust_thickness = st.sidebar.slider("Crust Thickness (Top Basement to Moho) (km)", mmd[0], mmd[1], mmd[2])
	crust_thickness *= 1000

	top_basement = int((geol_column[-2]) / 1000)

	moho_depth = deepcopy(int(top_basement + crust_thickness/1000))
	st.sidebar.subheader(f"Moho depth = {moho_depth} km")
	
	mmd = crust_dict[crust_type]["Upper Crust RHP"]

	rhp = st.sidebar.slider("Crust RHP (uW/m3)", mmd[0], mmd[1], mmd[2])
	rhp *= 1.0e-6

	uc_rhp = rhp / (0.5 * 0.33 + 0.67)
	lc_rhp = min(0.5 * uc_rhp, 0.4e-6)

	mmd = crust_dict[crust_type]["Mantle Thickness"]

	ast_depth = st.sidebar.slider("Depth to Astenosphere (km)", 20 + mmd[0], 40 + mmd[1], 20 + mmd[2])
	mantle_thickness = 1000*ast_depth - moho_depth*1000
	st.sidebar.subheader(f"Upper Mantle thickness = {int(mantle_thickness/1000)} km")
	
	geol_column.append(crust_thickness)
	geol_column.append(lc_rhp)
	geol_column.append(mantle_thickness)
	geol_column.append(uc_rhp)
	geol_column.append(present_day_temperature)

	##### RESHAPE
	geol_column = np.array(geol_column).reshape((49,1,1))
	#####

	display_mode = st.sidebar.selectbox("Maturity property", ["TMax", "EasyRo"])
	st.sidebar.latex(r"{\rm OrganoFacies}")

	of_select = st.sidebar.selectbox("Organofacies selection (For onset of Oil Window and STS-TMax correlation)", tmax_headers[1:])

	if display_mode == "TMax":
		tmax = tmax_of[tmax_headers[0]].values
		of_sts = tmax_of[of_select].values

	st.sidebar.latex(r"{\rm Options\, for\, display}")

	min_tmax = 400
	max_tmax = 600

	if display_mode == "TMax":
		min_tmax = float(st.sidebar.text_input("Minimum Tmax (C)", 400))
		max_tmax = float(st.sidebar.text_input("Maximum Tmax (C)", 600))
	
	elif display_mode == "EasyRo":
		min_Ro = float(st.sidebar.text_input("Minimum EasyRo (C)", 0.2))
		max_Ro = float(st.sidebar.text_input("Maximum EasyRo (C)", 4.0))

	min_depth = float(st.sidebar.text_input("Minimum depth (m)", 0.0))
	max_depth = float(st.sidebar.text_input("Maximum depth (m)", max(1000*top_basement + 1000, 7000)))
	max_temperature = float(st.sidebar.text_input("Maximum Present Day Temperature (C)", 400))

	st.sidebar.image('data/grey.png')

	temperature, maturity = predict(geol_column, neural_network_jarufah_basin)

	temperature = temperature[:,0,0].flatten()
	maturity = maturity[:,0,0].flatten()

	temperature = np.insert(temperature, 0, present_day_temperature)
	maturity = np.delete(maturity, [1,6]) #Delete values in hiatus
	temperature = np.delete(temperature, [1,6]) #Delete values in hiatus

	maturity = np.insert(maturity, 0, 0.2045)

	sts = np.interp(maturity/100, ro_sts['ezRo'], ro_sts['sts'])
	if display_mode == "TMax":
		tmax_d = np.interp(sts, of_sts, tmax)
	
	depths = geol_column[:43].flatten()
	mid_points = 0.5 * (depths[1:] + depths[:-1])
	mid_points = np.insert(mid_points, 0, 0)
	list_depths = [int(u) for u in depths[indx_to_keep]]

	onset_oil_window = lim_oil_sts[of_select]
	top_oil_window = np.interp(onset_oil_window, sts, mid_points)
	lim1 = np.interp(145, sts, mid_points)
	lim2 = np.interp(170, sts, mid_points)
	lim3 = np.interp(220, sts, mid_points)

	with st.expander("Summary"):
		st.markdown(summary_md)
		st.subheader("Location Map of the Jafurah basin and Ghawar field. Initial depths values in this app are estimated from Well E")
		st.text("Image from A. Hakami and S. Inan, 2016")
		st.image('data/Location_Map.png')

	user_data = st.file_uploader("Load your calibration data ! (Excel format, 5 columns : [depth, raw temperature, corrected temperature, TMax, easyRo])")
	data = None
	if user_data is not None:
		xls = pd.read_excel(user_data)
		data = xls.values
	colors = ['#FFF798', '#EEA26D', '#7FC06C', '#D5EDF5', '#C6E7F1', '#C6E7F1', '#BAE0E5', '#BAE0E5', '#72C0E0', '#804C8D', '#D75F49', '#7CB4B9']

	sts_map = create_sts_map(mid_points, sts, min_tmax, max_tmax)

	extent = [min_tmax, max_tmax, mid_points[0], mid_points[-1]]
	
	threshold_display = (max_depth - min_depth) / 70

	if display_mode == "TMax":
			caption = 'Computed Temperature and TMax profiles'
	else:
		caption = 'Computed Temperature and Easy Ro profiles'

	st.subheader(caption)
	with _lock:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,12))
		ax1.plot(temperature, mid_points, 'ko-', lw = 3)
		ax1.plot(sts, mid_points, 'o--', c='white')
		ax1.set_ylim([min_depth,max_depth])

		ax1.invert_yaxis()
		ax1.set_xlabel('Computed Temperature and STS (white curve) (C)')
		ax1.set_ylabel('Depth (MD m)')
		ax1.set_xlim([0,max_temperature])
		
		if data is not None:
			ax1.plot(data[:,1], data[:,0], 'ko')
			ax1.plot(data[:,2], data[:,0], 'ko')

		ax1.grid()
		markers = [([0,max_temperature],[depths[0], depths[0]])]

		idx=0
		for i in range(12):
			nb_marks = len(layers_wrap[i])
			idx += nb_marks
			markers.append(([0,max_temperature],[depths[idx], depths[idx]]))

		for ii,m in enumerate(markers):
			skip = False
			ax1.plot(m[0], m[1], 'k--', lw=0.8)
			x1 = m[0][0]
			y1 = m[1][0]
			x2 = m[0][1]
			y2 = m[1][1]
			if ii < len(markers) - 1:
				y3 = markers[ii+1][1][1]
				y4 = markers[ii+1][1][0]
			else:
				y3 = y2
				y4 = y1

			y = np.array([[x1,y1], [x2, y2], [x2, y3], [x1, y4]])
			if ii > 0 and ii < 12:
				alpha = 0.5
				p = Polygon(y, facecolor = colors[ii], alpha = alpha)
				ax1.add_patch(p)
				if ii > 1 and m[1][0] - markers[ii-1][1][0] > threshold_display:
					if 'Hanifa' in layers_dict[ii] or 'Jubaila' in layers_dict[ii]:
						layers_dict[ii] += ' Source Rock'
						ax1.annotate(f"{layers_dict[ii]} - {int(list_depths[ii])}m", (max_temperature, m[1][0] + threshold_display/2), ha = 'right', bbox={'facecolor': 'white', 'edgecolor':'none', 'alpha': 0.5, 'pad': 1})
					else:
						ax1.annotate(f"{layers_dict[ii]} - {int(list_depths[ii])}m", (max_temperature, m[1][0] + threshold_display), ha = 'right')
		ax1.annotate(f"Top Basement - {int(list_depths[12])}m", (max_temperature, m[1][1] + 150), ha = 'right')

		if display_mode == "EasyRo":
			max_point = max_Ro
			min_point = min_Ro
			ax2.plot(maturity, mid_points, 'o-', c='black')
			if data is not None:
				ax2.plot(data[:,4], data[:,0], 'ko')

			ax2.set_xlabel('Easy Ro (%Ro eq.)')

		elif display_mode == "TMax":
			max_point = max_tmax
			min_point = min_tmax
			ax2.plot(tmax_d, mid_points, 'o-', c='black')
			if data is not None:
				ax2.plot(data[:,3], data[:,0], 'ko')

			ax2.set_xlabel('Computed TMax (C)')

		ax2.set_ylim([min_depth,max_depth])
		ax2.invert_yaxis()
		ax2.set_ylabel('Depth (MDm)')
		ax2.set_xlim([min_point,max_point])
		ax2.grid()

		# if st.session_state.of_select != of_select:
		# 	colormap_display = get_cmp(of_select)
		# 	st.session_state.of_select = of_select
		colormap_display = get_cmp(of_select)

		ax2.imshow(sts_map, extent=[min_point,max_point, 0,mid_points[-1]], cmap=colormap_display, origin='lower', aspect='auto', vmin=90, vmax=250)

		markers = [([min_point,max_point],[depths[0], depths[0]])]
		idx=0
		for i in range(12):
			nb_marks = len(layers_wrap[i])
			idx += nb_marks
			markers.append(([min_point,max_point],[depths[idx], depths[idx]]))
		
		for ii,m in enumerate(markers):
			ax2.plot(m[0], m[1], 'k--', lw=0.5)

		if mid_points[-1] - top_oil_window > 100 and of_select != 'F':
			ax2.annotate('Early Oil  ', (max_point, top_oil_window + 150), ha = 'right')

		if lim1 - top_oil_window > 200 and mid_points[-1] - lim1 > 100 and of_select != 'F':
			ax2.annotate('Volatile Oil  ', (max_point, lim1 + 25), ha = 'right')

		if lim2 - lim1 > 200 and mid_points[-1] - lim2 > 100:
			ax2.annotate('Gas window  ', (max_point, lim2 + 25), ha = 'right')

		if lim3 - lim2 > 200 and mid_points[-1] - lim3 > 100:
			ax2.annotate('Floor Gas Window  ', (max_point, lim3 + 150), ha = 'right')

		buf = BytesIO()
		fig.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
		
		st.image(buf, use_column_width=False, caption=caption)

	st.subheader("Coming soon : burial history and expulsion history plots !")
	
if __name__ == "__main__":
	st_ui()