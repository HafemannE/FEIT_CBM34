
with open('module1_mesh.py','r') as mesh_file:
    mesh_code=mesh_file.readlines()
    
with open('module2_forward.py','r') as forward_file:
    forward_code = forward_file.readlines()
    forward_code[0]="\n"
    forward_code[1]="\n"

with open('module3_inverse.py','r') as inverse_file:
    inverse_code = inverse_file.readlines()
    inverse_code[0]="\n \n"

with open('module4_auxiliar.py','r') as module4_file:
    auxiliar_code = module4_file.readlines()
    auxiliar_code[0]="\n"
    auxiliar_code[1]="\n"

with open("FEIT_onefile.py", "w") as f:
    #for code in [mesh_code]:#, forward_code, inverse_code, auxiliar_code]:
        #for line in code:
    f.writelines(mesh_code)
    f.writelines(forward_code)
    f.writelines(inverse_code)
    f.writelines(auxiliar_code)
    f.close()