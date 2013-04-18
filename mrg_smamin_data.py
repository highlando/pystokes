import plot_utils as plu
tol = 2**(-10)
HL = [16,32,64,128,256,512,1024]

#plu.merge_json_dicts(    'json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[0] + 'to%d' % HL[0] + 'Mesh40HalfExpEulSmaMin.json',
#					     'json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[3] + 'to%d' % HL[1] + 'Mesh40HalfExpEulSmaMin.json')

#plu.merge_json_dicts(    'json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[2] + 'to%d' % HL[2] + 'Mesh40HalfExpEulSmaMin.json',
#					     'json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[3] + 'to%d' % HL[3] + 'Mesh40HalfExpEulSmaMin.json')

plu.merge_json_dicts(    'json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[4] + 'to%d' % HL[4] + 'Mesh40HalfExpEulSmaMin.json',
					     'json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[5] + 'to%d' % HL[5] + 'Mesh40HalfExpEulSmaMin.json')

#plu.merge_json_dicts('json/MrgdOmeg8Tol%1.2e' % tol + 'NTs%d' % HL[0] + 'to%d' % HL[1] + 'Mesh40HalfExpEulSmaMin.json',
#					 'json/MrgdOmeg8Tol%1.2e' % tol + 'NTs%d' % HL[2] + 'to%d' % HL[3] + 'Mesh40HalfExpEulSmaMin.json')

plu.merge_json_dicts('json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[0] + 'to%d' % HL[3] + 'Mesh40HalfExpEulSmaMin.json',
					 'json/MrgdOmeg8Tol%1.2e' % tol + 'NTs%d' % HL[4] + 'to%d' % HL[5] + 'Mesh40HalfExpEulSmaMin.json')

plu.merge_json_dicts('json/MrgdOmeg8Tol%1.2e' % tol + 'NTs%d' % HL[0] + 'to%d' % HL[5] + 'Mesh40HalfExpEulSmaMin.json',
					 'json/Omeg8Tol%1.2e' % tol + 'NTs%d' % HL[6] + 'to%d' % HL[6] + 'Mesh40HalfExpEulSmaMin.json')

plu.convpltjsd('json/MrgdOmeg8Tol%1.2e' % tol + 'NTs%d' % HL[0] + 'to%d' % HL[6] + 'Mesh40HalfExpEulSmaMin.json')



