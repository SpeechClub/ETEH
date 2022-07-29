import sys

if __name__=="__main__":
    rfile = sys.argv[1]
    with open(rfile,'r',encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for l in lines:
        nl = l.replace('▁',' ')
        if nl[0] == ' ':
            nl = nl[1:]
        new_lines.append(nl)
    with open(rfile,'w',encoding='utf-8') as f:
        f.writelines(new_lines)
