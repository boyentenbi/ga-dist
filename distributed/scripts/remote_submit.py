import paramiko

def ssh_command(ssh):
    command = input("Command:")
    ssh.invoke_shell()
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read())

def ssh_connect(host, user):
    try:
        ssh = paramiko.SSHClient()
        print('Calling paramiko')
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=user)

        ssh_command(ssh)
    except Exception as e:
        print('Connection Failed')
        print(e)

if __name__=='__main__':


    host = 'login-cpu.hpc.cam.ac.uk'
    user = 'pc517'

    password = input("Password:")
    ssh_connect(host, user, password = password)



    ssh.close()