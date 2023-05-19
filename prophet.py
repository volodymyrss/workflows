import glob
import os
import datetime
import pandas as pd
from io import StringIO

def gt(dt_str):
    dt, _, us= dt_str.partition(".")
    
    dt = dt.rstrip("Z")
        
    dt= datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    
    if len(us)>0:
        us= int(us.rstrip("Z"), 10)
        dt += datetime.timedelta(microseconds=us)
    return dt

arc_root_prefix="/mnt/sshfs/isdc-in01"

def read_tsf(rev, debug=False):

    patt=arc_root_prefix+"/isdc/pvphase/nrt/ops/aux/org/%.4i/TSF_*.INT"%rev

    print("arcroot prefix:", arc_root_prefix)
    print("patt", patt)
        
    if debug:
        print(patt)

    tsfs = glob.glob(patt)

    backup_tsf = "/cdci-resources/tsfs/tsf_%i"%rev
    if len(tsfs)==0 and os.path.exists(backup_tsf):
        tsfs=[backup_tsf]
    
    print("all tsfs", tsfs)

    tsfinfo = {}
    if len(tsfs) >= 1:
        tsf = tsfs[0]

        s = StringIO()
        for l in open(tsf):
            if 'COMMENT' not in l:
                s.write(l)
        s.seek(0)

        d = pd.read_fwf(s, colspecs=[(0,20), (20,42), (42,63), (63,200)],
                        names=['time', 'name', 'value', 'comment'],skiprows=8)

        print("read", d)


        #d['time'] = list(map(gt,d['time']))
        #['name'] = [s.strip() for s in d['name']]
    else:
        print("unabel to find tsp for",rev,"all tsfs",tsfs)
        return 
    
    starts = []
    for start_marker in 'SPI_START', 'IBIS_START':
        starts += list(d[d['name'] == start_marker]['time'])
        
        
    stops = []
    for stop_marker in 'SPI_STOP', 'IBIS_STOP':
        stops += list(d[d['name'] == stop_marker]['time'])
        
    return dict(        
        start = sorted(list(map(gt, starts)))[0],
        stop = sorted(list(map(gt, stops)))[-1],
    )





def predict(rev=None, time=None, debug=False):
    if time is None:
        time = datetime.datetime.utcnow()
    else:
        if not isinstance(time, datetime.datetime):
            time = gt(time)

    if rev is None:
        import integralclient as ic
        rev = int(ic.converttime("UTC", time.strftime("%Y-%m-%dT%H:%M:%S"), "REVNUM"))

    prevrev_boundaries_tsf = read_tsf(rev-1)
    thisrev_boundaries_tsf = read_tsf(rev)
    nextrev_boundaries_tsf = read_tsf(rev+1)


    promise = ""

    last_data_utc=None
    next_data_utc=None
    expected_data_status=None

    if time < thisrev_boundaries_tsf['start']:
        if debug:
            print('out of rev')
            
            print("last data %.3lg"%((time-prevrev_boundaries_tsf['stop']).total_seconds()/3600.))
            print("next data %.3lg"%((thisrev_boundaries_tsf['start']-time).total_seconds()/3600.))    
        
        promise = "break since %.2lg hr next data in %.2lg hr"%(
            (time - prevrev_boundaries_tsf['stop']).total_seconds()/3600.,
            (thisrev_boundaries_tsf['start']-time).total_seconds()/3600.,
                  )

        last_data_utc = prevrev_boundaries_tsf['stop'].strftime("%Y-%m-%dT%H:%M:%S")
        next_data_utc = thisrev_boundaries_tsf['start'].strftime("%Y-%m-%dT%H:%M:%S")

        expected_data_status = "OFFLINE"

    elif time > thisrev_boundaries_tsf['start'] and time < thisrev_boundaries_tsf['stop']:
        if debug:
            print("in rev data")
        if nextrev_boundaries_tsf  is not None:
            promise = "next break in data in %.2lg hr: %s, for %.2lg hr" %(
                    (thisrev_boundaries_tsf['stop'] - time).total_seconds()/3600,
                    thisrev_boundaries_tsf['stop'].strftime("%Y-%m-%dT%H:%M:%S"),
                    (nextrev_boundaries_tsf['start'] - thisrev_boundaries_tsf['stop']).total_seconds()/3600,
                )
        else:
            promise = "next break in data in %.2lg hr: %s, for about 9 hr" %(
                    (thisrev_boundaries_tsf['stop'] - time).total_seconds()/3600,
                    thisrev_boundaries_tsf['stop'].strftime("%Y-%m-%dT%H:%M:%S"),
                )

        expected_data_status = "ONLINE"

    elif time > thisrev_boundaries_tsf['stop']:
        if debug:
            print('after rev')
            print("last data %.3lg"%((time-thisrev_boundaries_tsf['stop']).total_seconds()/3600.))
        
        if nextrev_boundaries_tsf['start']>time:
            if debug:
                print("next data expected %.2lg"%((nextrev_boundaries_tsf['start']-time).total_seconds()/3600.))
            promise = "break since %.2lg hr next data in %.2lg hr"%(
                            (time - thisrev_boundaries_tsf['stop']).total_seconds()/3600.,
                            (nextrev_boundaries_tsf['start']-time).total_seconds()/3600.,
                        )

            last_data_utc = thisrev_boundaries_tsf['stop'].strftime("%Y-%m-%dT%H:%M:%S")
            next_data_utc = nextrev_boundaries_tsf['start'].strftime("%Y-%m-%dT%H:%M:%S")
            expected_data_status = "OFFLINE"
        else:
            print("current rev is not current?")

    if debug:
        print("PROMISE: ",promise)

    summary = dict(
        prevrev=dict(
            num=rev-1,
            start=prevrev_boundaries_tsf['start'].strftime("%Y-%m-%dT%H:%M:%S"),
            stop=prevrev_boundaries_tsf['stop'].strftime("%Y-%m-%dT%H:%M:%S"),
        ),
        thisrev=dict(
            num=rev,
            start=thisrev_boundaries_tsf['start'].strftime("%Y-%m-%dT%H:%M:%S"),
            stop=thisrev_boundaries_tsf['stop'].strftime("%Y-%m-%dT%H:%M:%S"),
        ),
        nextrev=dict(
            num=rev+1,
            start=nextrev_boundaries_tsf['start'].strftime("%Y-%m-%dT%H:%M:%S"),
            stop=nextrev_boundaries_tsf['stop'].strftime("%Y-%m-%dT%H:%M:%S"),
        ) if nextrev_boundaries_tsf is not None else None,
        last_data_utc=last_data_utc,
        next_data_utc=next_data_utc,
        expected_data_status=expected_data_status,
    )
                   
    return promise, summary
