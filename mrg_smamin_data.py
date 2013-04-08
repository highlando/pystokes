tol = '04'

plu.merge_json_dicts(    'json/Omeg8Tol1e-' + tol + 'NTs16to16Mesh40HalfExpEulSmaMin.json',
					     'json/Omeg8Tol1e-' + tol + 'NTs32to32Mesh40HalfExpEulSmaMin.json')

plu.merge_json_dicts(    'json/Omeg8Tol1e-' + tol + 'NTs64to64Mesh40HalfExpEulSmaMin.json',
					     'json/Omeg8Tol1e-' + tol + 'NTs128to128Mesh40HalfExpEulSmaMin.json')

#plu.merge_json_dicts(    'json/Omeg8Tol1e-' + tol + 'NTs256to256Mesh40HalfExpEulSmaMin.json',
#					     'json/Omeg8Tol1e-' + tol + 'NTs512to512Mesh40HalfExpEulSmaMin.json')

plu.merge_json_dicts('json/MrgdOmeg8Tol1e-' + tol + 'NTs16to32Mesh40HalfExpEulSmaMin.json',
					 'json/MrgdOmeg8Tol1e-' + tol + 'NTs64to128Mesh40HalfExpEulSmaMin.json')

#plu.merge_json_dicts('json/MrgdOmeg8Tol1e-' + tol + 'NTs16to128Mesh40HalfExpEulSmaMin.json',
#					 'json/MrgdOmeg8Tol1e-' + tol + 'NTs256to512Mesh40HalfExpEulSmaMin.json')

plu.convpltjsd('json/MrgdOmeg8Tol1e-' + tol + 'NTs16to128Mesh40HalfExpEulSmaMin.json')



