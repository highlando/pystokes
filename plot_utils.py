import json
import matplotlib.pyplot as plt

def load_json_dicts(StrToJs):

	fjs = open(StrToJs)
	JsDict = json.load(fjs)
	return JsDict


def merge_json_dicts(CurDi,DiToAppend):

	Jsc = load_json_dicts(CurDi)
	Jsa = load_json_dicts(DiToAppend)

	if Jsc['SpaceDiscParam'] != Jsa['SpaceDiscParam'] or Jsc['Omega'] != Jsa['Omega']:
		raise Warning('Space discretization or omega do not match')

	Jsc['TimeDiscs'].extend(Jsa['TimeDiscs'])
	Jsc['ContiRes'].extend(Jsa['ContiRes'])
	Jsc['VelEr'].extend(Jsa['VelEr'])
	Jsc['PEr'].extend(Jsa['PEr'])
	#Jsc['TolCor'].extend(Jsa['TolCor'])

	JsFile = 'json/MrgdOmeg%dTol%0.2eNTs%dto%dMesh%d' % (Jsc['Omega'], Jsc['LinaTol'], Jsc['TimeDiscs'][0], Jsc['TimeDiscs'][-1], Jsc['SpaceDiscParam']) +Jsc['TimeIntMeth'] + '.json'

	f = open(JsFile, 'w')
	f.write(json.dumps(Jsc))

	print 'Merged data stored in \n("' + JsFile + '")'

	return 

def convpltjsd(Jsc):

	Jsc = load_json_dicts(Jsc)
	
			
	Mdict = {'HalfExpEulInd2': 'Ind2', 'HalfExpEulSmaMin': 'Ind1', 'Heei2Ra':'Ind2ra'}
	JsFile = 'om%d' % Jsc['Omega'] + 'json/' + Mdict[Jsc['TimeIntMeth']] + 'Tol%1.1eN%d' % (Jsc['LinaTol'], Jsc['SpaceDiscParam']) + '.json'

	f = open(JsFile, 'w')
	f.write(json.dumps(Jsc))

	print 'Data stored in \n("' + JsFile + '")'

	return
	
def jsd_plot_errs(JsDict):

	JsDict = load_json_dicts(JsDict)

	plt.close('all')
	for i in range(len(JsDict['TimeDiscs'])):
		leg = 'NTs = $%d$' % JsDict['TimeDiscs'][i]
		plt.figure(1)
		plt.plot(JsDict['ContiRes'][i],label=leg)
		plt.title(JsDict['TimeIntMeth']+': continuity eqn residual')
		plt.legend()
		plt.figure(2)
		plt.plot(JsDict['VelEr'][i],label=leg)
		plt.title(JsDict['TimeIntMeth']+': Velocity error')
		plt.legend()
		plt.figure(3)
		plt.plot(JsDict['PEr'][i],label=leg)
		plt.title(JsDict['TimeIntMeth']+': Pressure error')
		plt.legend()

	plt.show()

	return
