Ebmin = 20
Ebmax = 100

PVmin = 0
PVmax = 1.111

Dmin = 2
Dmax = 340

n_bins = 10

p = 22
learning_rate = 1e-3
cost_lr = 1e-4
eps = 0.5
discount_factor = 0.9
num_days = 1095
verbose_ = 1

initial_state = (PVmin, Ebmax, Dmin)
initial_action = 2 # (0 - charging, 1 - discharging, 2 - Nothing)

demand_csv_path = 'Data/files/demand.csv'
data_csv_path = 'Data/files/solar.csv'
final_csv_path = 'Data/files/final_csv.csv'

q_table_path = 'Data/weights/power system agent q learning.npy'
cum_cost_path = 'Data/visualization/power_system_agent.png'
Egrid_path = 'Data/visualization/E grid effiency.png'
Eb_path = 'Data/visualization/E battery effiency.png'
data_columns = ['Month','Day','Hour','PV_component','Demand']

#Solar DNN estimator
solar_dnn_csv = 'Data/files/solar_forecasting.csv'
solar_dnn_cols = ['Wind Speed (m/s)','Plane of Array Irradiance (W/m^2)','Cell Temperature (C)','AC System Output (W)']
dim1 = 32
dim2 = 1
batch_size = 64
validation_split = 0.2
verbose=1
epochs = 100
solar_weights = 'Data/weights/solar_model.h5'
loss_img = 'Data/visualization/solar_loss.png'