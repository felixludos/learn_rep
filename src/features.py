
import omnifig as fig

@fig.Script('extract-features', description='Extract features from image datasets using pretrained models')
def extract_features(A):
	dataset = fig.run('load-data', A)
	
	model = fig.run('load-model', A)

	