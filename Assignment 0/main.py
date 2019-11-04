from pattern import *
from generator import *


'''
Checker
'''
sample_checker = Checker(250,25)
sample_checker.draw()
sample_checker.show()

'''
Spectrum
'''

sample_spectrum = Spectrum(256)
sample_spectrum.draw()
sample_spectrum.show()

'''
Circle
'''

sample_circle = Circle(250, 50, (130,70))
sample_circle.draw()
sample_circle.show()

#########################
'''
Generator
'''

sample_ImageGenerator = ImageGenerator('exercise_data/', 'Labels.json', 8, [256,256,3], True, True, True)
sample_ImageGenerator.show()
# sample_ImageGenerator.show()