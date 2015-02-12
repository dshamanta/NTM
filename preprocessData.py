import os, fnmatch, re

def findfiles (path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)

with open("ntm_3group_news.txt", "w+") as fout:
    cnt = 0
    for textfile in findfiles(r'./3GroupsFrom20News', '*'):
        if not textfile.startswith(r'./3GroupsFrom20News\\1'):
            print textfile
            with open (textfile, "r") as myfile:
                data=myfile.read().replace('\n', ' ').lower()
                data = re.sub(r'[^a-zA-Z0-9 ]', '', data)
            cnt = cnt + 1
            if(cnt % 100 == 0):
                print("File Processed: %d"%cnt)
            fout.write(data + '\n')
    ##        print filename
    print ('Cnt = %d'%cnt)



##lst = [x[0] for x in os.walk('./3GroupsFrom20News')]
##
##for subfolder in lst:
##    fileCount =  len([name for name in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, name))])
##    print ('%s\t%d'%(subfolder, fileCount))
