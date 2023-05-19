import os


def get_arc_root_prefix():
    arc_root_prefix = None
    arc_root_prefixes = ["/mnt/sshfs/isdc-in01/","/", os.path.join(os.environ.get("HOME"),"scratch/data/integral-nrt")]

    for arc_root_prefix in arc_root_prefixes:
        if os.path.exists(arc_root_prefix+"/isdc/arc/rev_3/idx/scw/GNRL-SCWG-GRP-IDX.fits"):
            print("picking this arc root prefix:",arc_root_prefix)
            break
        
    if arc_root_prefix is None:
        raise Exception("no archive found!")

    return arc_root_prefix
