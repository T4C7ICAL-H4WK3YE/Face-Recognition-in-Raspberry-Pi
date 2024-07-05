import pickle
import os

# Path to the encodings pickle file
encodingsP = "C:/Users/Dell/Pictures/Camera Roll/Dataset/encodings_new.pickle"

# Load existing encodings
if os.path.exists(encodingsP):
    with open(encodingsP, 'rb') as f:
        data = pickle.load(f)
    knownEncodings = data['encodings']
    knownNames = data['names']
    print("[INFO] existing encodings loaded.")
else:
    print("[ERROR] pickle file not found.")
    exit()

# Function to remove specific name and its corresponding encodings
def remove_name_from_encodings(name_to_remove):
    global knownEncodings, knownNames
    updated_encodings = []
    updated_names = []
    removed = False
    
    for encoding, name in zip(knownEncodings, knownNames):
        if name == name_to_remove:
            removed = True
            print(f"[INFO] Removing {name_to_remove} from encodings.")
        else:
            updated_encodings.append(encoding)
            updated_names.append(name)
    
    if not removed:
        print(f"[INFO] {name_to_remove} not found in encodings.")

    return updated_encodings, updated_names

# Specify the name to remove
name_to_remove = "Mr. Prakash"

# Remove the specific name and its corresponding encodings
knownEncodings, knownNames = remove_name_from_encodings(name_to_remove)

# Save the updated encodings back to the pickle file
print("[INFO] serializing updated encodings...")
data = {'encodings': knownEncodings, 'names': knownNames}
with open(encodingsP, 'wb') as f:
    f.write(pickle.dumps(data))
print("[INFO] updated encodings serialized to disk.")
