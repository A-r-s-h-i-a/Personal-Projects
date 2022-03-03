# Music Generation
A music generation model was created using sequence networks, LSTMs, combined with traditional neural networks. Notes from songs were passed through the model, where it was the model's objective to guess the next note when given the previous 'n' notes as input. The model was made polyphonic by having the melody and harmony passed to two inputs in parallel, then having these data-streams combined internally to the model so that independent and dependent associations between them could be drawn. Data preprocessing techniques were employed to make all this easier and improve learning. These techniques were mostly music-theory related, and included things like transposing all songs in major-keys to C-major and all songs in minor-keys to A-minor for reducing unwanted dataset dimensionality.

The dataset used is purposefully missing from this directory, but it can all be found here: https://github.com/jukedeck/nottingham-dataset

# Result
[Sample Audio Output](https://github.com/A-r-s-h-i-a/Personal-Projects/blob/main/Music%20Generation/PolyphonicArchitecture_Test1.mp3)


https://user-images.githubusercontent.com/46332063/156494861-c14ba73b-0610-4920-8a8f-7412be07daed.mp4

