# Utilities for the instagram bot
# Various globals and the like
import torch



USE_GPU = True if torch.cuda.is_available() else False
SAVE_IMG_PATH = "Generated_Images"
NUM_IMAGES = 100
MODEL = "celebAHQ-512" # Low Quality=="celeba"  High Quality=="celebAHQ-512"
IMG_CAPTIONS = [
	'Have you seen me?',
	'Have you seen this person?',
	'Please let me know if you have seen this person!',
	'Any chance you have seen this person?',
	'Where have you seen this person?',
	'Does this person look familiar to you?',
	'Do you think you have seen this person?',
	'Any chance you\'ve seen this person?',
	'Where have you seen me?',
	'Please let me know if you have seen me.',
	'Do you think you have seen this person from a distance?',
	'Any chance you have seen me from a distance?',
	'Please tell me if you\'ve seen me!',
	'Do you think you have seen me from afar?',
	'Have you or anyone you know ever seen me?',
	'Have you or your friends ever caught sight of me?',
	'Do you believe you\'ve caught a glimpse of me before?',
	'Any of your friends or you know who this is?',
	'If you know who this is, could you please let me know?',
	'If you have seen this person recently, could you please let me know?',
	'Where have you seen me?',
	'Is there any chance you have seen this person!',
	'Is there any chance you or your friends have seen this person?',
	'Any way you\'ve seen this person? From a distance maybe?',
	'Could you let me know if you\'ve seen this person! Thank you!',
]