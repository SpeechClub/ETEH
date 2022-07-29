# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
def char_listreader(path, sc=' ',append=True, insert=True, eos='<eos>'):
    with open(path,'r',encoding='utf-8') as f:
        char_list = f.read().splitlines()
    for i in range(len(char_list)):
        if isinstance(char_list[i],str):
            char_list[i] = char_list[i].split(sc)[0]
        else:
            char_list[i] = str(char_list[i])
    if insert:
        char_list.insert(0,'<blank>')
    if append and eos not in char_list:
        char_list.append('<eos>')
    return char_list

def dict_reader(path,sc=' ',append=True, eos='<eos>'):
    world_dict = {}
    last = 0
    with open(path,'r',encoding='utf-8') as f:
        lines = f.read().splitlines()
    for line in lines:
        key, value = line.split(sc)[0], int(line.split(sc)[1])
        world_dict[key] = value
        last = value + 1
    if append:
        world_dict[eos] = last
    return world_dict

def txt_reader(path, sc=' ', bc=0, id_dict=None):
    line_lists = []
    with open(path,'r',encoding='utf-8') as f:
        line = f.readline()
        while True:
            if len(line)>=1:
                line=line[:-1]
                line_l = line.split(sc)[bc:]
                while '' in line_l:
                    line_l.remove('')
                if id_dict is not None:
                    id_line = [id_dict[x] for x in line_l]
                else: 
                    id_line = line_l
                line_lists.append(id_line)            
            line = f.readline()
            if line == "":
                break
    return line_lists