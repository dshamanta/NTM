import pickle

# Saving the objects:
with open('filename.pk', 'w') as f:
    pickle.dump([model.w_ld, model.w_lt], f)

# Getting back the objects:
ld = []
lt = []
with open('filename.pk') as f:
    ld, lt = pickle.load(f)
