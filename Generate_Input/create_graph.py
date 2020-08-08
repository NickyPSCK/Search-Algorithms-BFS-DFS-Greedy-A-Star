import pandas as pd
from math import cos, asin, sqrt, pi

# ------------------------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------------------------
end_state = 'Florida'
output_file = 'in_state.txt'
# ------------------------------------------------------------------------------------------------

def distance(lat1, lon1, lat2, lon2, decimal):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return round(12742 * asin(sqrt(a)), decimal) # Km

# Load Raw Data
data = pd.read_csv('raw_state_graph/latlong.txt')
with open('raw_state_graph/relation.txt', 'r') as f:
    relations = f.readlines()

data.set_index('state_no_space', inplace=True)

lines_to_write = []
for relation in relations:
    line_list = relation.strip().split()

    line_to_write = ''
    for i, state in enumerate(line_list):

        if i==0:
            line_to_write += state + ' ' + str(int(distance(data.loc[state,'Lat'], 
                                                        data.loc[state,'Long'], 
                                                        data.loc[end_state,'Lat'], 
                                                        data.loc[end_state,'Long'],
                                                        0)))
        else:
            line_to_write += ' ' + state + ' ' + str(distance(data.loc[line_list[0],'Lat'], 
                                                                    data.loc[line_list[0],'Long'], 
                                                                    data.loc[state,'Lat'], 
                                                                    data.loc[state,'Long'],
                                                                    2))
    lines_to_write.append(line_to_write+'\n')
lines_to_write.insert(0, end_state+'\n')                                                                

# Write input file
with open(output_file, 'w') as f:
    f.writelines(lines_to_write)
