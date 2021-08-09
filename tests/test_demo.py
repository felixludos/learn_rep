
import omnifig as fig



def test_demo():
	fig.initialize()

	A = fig.get_config('demo')

	A.push('budget', 10)
	fig.run(None, A)




