import os

def BreadthFirstFileScan( root ) :
    dirs = [root]
    # while we have dirs to scan
    while len(dirs) :
        nextDirs = []
        for parent in dirs :
            # scan each dir
            for n in os.listdir( parent ) :
                
                h = os.path.join( parent, n )
                if os.path.isdir( h ) :
                    
                        nextDirs.append( h )
                if h.endswith(".txt"):
                    yield h
        
        dirs = nextDirs


def bfscan( path ) :
    for n in BreadthFirstFileScan( path ) :
        print( n)


bfscan("C:")



