from mpi4py import MPI
from tensorflow.keras.models import load_model
import numpy as np
import io
import glob

valid_dir = "C:\Users\renji\Downloads\archive\C-NMC_Leukemia\validation_data\C-NMC_test_prelim_phase_data"
data_valid = []
file_path = "C:\Users\renji\Downloads\archive\C-NMC_Leukemia\validation_data\C-NMC_test_prelim_phase_data_labels.csv"
column_name = 'labels'
label_valid = []
valid_imglist = glob.glob(valid_dir+'/*')
for img in valid_imglist[:]:
  image = cv2.imread(img)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  data_valid.append(image)

with open(file_path, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    column_index = header.index(column_name)
    for row in csvreader:
        label_valid.append(row[column_index])
        
print(len(data_valid))
print(len(label_valid))
print(label_valid[0:5])
data = data_valid[:300]
test_label = label_valid[:300]
print(len(data))
print(len(test_label))
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processes = comm.Get_size()

# Load the model
if rank == 0:
    # Load your model here
    model_file = "resnet_leukemia_model.h5"
    with open(model_file, "rb") as model_file:
        model_data = model_file.read()
    loaded_model = load_model(io.BytesIO(model_data))
    comm.bcast(loaded_model, root=0)
else:
    loaded_model = comm.bcast(None, root=0)

# Distribute data across processes
data = comm.bcast(data, root=0)
total_images = len(data)
images_per_process = total_images // num_processes
start_index = rank * images_per_process
end_index = (rank + 1) * images_per_process
data_slice = data[start_index:end_index]

# Perform predictions for each data point in parallel
predictions = []
for data_point in data_slice:
    prediction = loaded_model.predict(np.expand_dims(data_point, axis=0))
    predictions.append(prediction)

# Gather all predictions to the root process
all_predictions = comm.gather(predictions, root=0)

# On the root process, consolidate the predictions
if rank == 0:
    consolidated_predictions = []
    for process_predictions in all_predictions:
        consolidated_predictions.extend(process_predictions)

print(consolidated_predictions)
    # Now 'consolidated_predictions' contains predictions for all data points
