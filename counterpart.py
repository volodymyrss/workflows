from __future__ import print_function

import sys
import os
import string
import sys
import gzip
import time
import hashlib
import requests

import workflows

from scipy.stats import norm

import healpy

import plot
import healtics
import astropy.io.fits as pyfits
import json

from astropy.coordinates import SkyCoord, PhysicsSphericalRepresentation
from astropy import units as u

import integralvisibility

from io import BytesIO
from astropy.io import fits

import integralclient as ic

from dataanalysis import core as da
from dataanalysis import hashtools
from dataanalysis.hashtools import shhash
import dataanalysis

import numpy as np
import integralclient

import healpy
import healtics
from scipy import stats
from matplotlib import pylab as p


def transform_rmap(rmap):
    nside = healpy.npix2nside(rmap.shape[0])
    npx = np.arange(healpy.nside2npix(nside))
    theta, phi = healpy.pix2ang(nside, npx)
    # SkyCoord(phi, theta, 1, unit=(u.rad, u.rad), representation="physicsspherical")
    ntheta = np.pi - theta
    nphi = np.pi + phi  # + or -???
    nphi[nphi > np.pi * 2] -= np.pi * 2
    return healpy.get_interp_val(rmap, ntheta, nphi)


def healpix_fk5_to_galactic(mp):
    nside = healpy.npix2nside(mp.shape[0])
    theta, phi = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    ra = phi / np.pi * 180
    dec = 90 - theta / np.pi * 180
    coord_galactic_gmap = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="galactic")
    coord_galactic_map = coord_galactic_gmap.transform_to("fk5")
    return healpy.get_interp_val(mp, coord_galactic_map.ra.degree, coord_galactic_map.dec.degree, lonlat=True)

def healpix_galactic_to_fk5(mp):
    nside = healpy.npix2nside(mp.shape[0])
    theta, phi = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    ra = phi / np.pi * 180
    dec = 90 - theta / np.pi * 180
    coord_galactic_gmap = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="fk5")
    coord_galactic_map = coord_galactic_gmap.transform_to("galactic")
    return healpy.get_interp_val(mp, coord_galactic_map.l.degree, coord_galactic_map.b.degree, lonlat=True)


def grid_for_healpix_map(mp):
    return healpy.pix2ang(healpy.npix2nside(mp.shape[0]),np.arange(mp.shape[0]),lonlat=True)

class CacheCounterparts(dataanalysis.caches.cache_core.CacheNoIndex):
    def get_burst(self, hashe):
        if isinstance(hashe, tuple):
            if hashe[0] == "analysis":
                if hashe[2].startswith('Event'):
                    return hashe[2]
                return self.get_burst(hashe[1])
            if hashe[0] == "list":  # more universaly
                for k in hashe[1:]:
                    r = self.get_burst(k)
                    if r is not None:
                        return r
                return None
            raise Exception("unknown tuple in the hash:" + str(hashe))
        if hashe is None:
            return None  # 'Any'
        if isinstance(hashe, str):
            return None
        raise Exception("unknown class in the hash:" + str(hashe))

    def get_rev(self, hashe):
        # if dataanalysis.printhook.global_log_enabled: print("search for rev in",hashe)
        if isinstance(hashe, tuple):
            if hashe[
                0] == "analysis":  # more universaly
                if hashe[2].startswith('Revolution'):
                    return hashe[1]
                return self.get_rev(hashe[1])
            if hashe[
                0] == "list":  # more universaly
                for k in hashe[1:]:
                    r = self.get_rev(k)
                    if r is not None:
                        return r
                return None
            raise Exception("unknown tuple in the hash:" + str(hashe))
        if hashe is None:
            return None
        if isinstance(hashe, str):
            return None
        raise Exception("unknown class in the hash:" + str(hashe))

    def hashe2signature(self, hashe):
        scw = self.get_burst(hashe)
        if scw is not None:
            if isinstance(hashe, tuple):
                if hashe[0] == "analysis":
                    return hashe[2] + ":" + scw + ":" + shhash(hashe)[:8]
            return shhash(hashe)[:8]

        return hashe[2] + ":" + shhash(hashe)[:16]

    def construct_cached_file_path(self, hashe, obj=None):
        # print("will construct INTEGRAL cached file path for",hashe)

        burst = self.get_burst(hashe)

        def hash_to_path2(hashe):
            return shhash(repr(hashe[1]))[:8]

        if not isinstance(burst, str):
            burst = None

        if burst == "Any":
            burst = None

        rev = self.get_rev(hashe)
        print("Rev in hashe:",rev)

        if burst is None or len(burst.split(".",2))!=3:
            if dataanalysis.printhook.global_log_enabled: print("not burst-grouped cache")

            if rev is None:
                r = self.filecacheroot + "/global/" + hashe[2] + "/"  + hash_to_path2(hashe) + "/"
            else:
                if dataanalysis.printhook.global_log_enabled: print("is rev-grouped cache")
                hashe = hashtools.hashe_replace_object(hashe, rev, "any")
                r = self.filecacheroot + "/byrev/" + rev + "/" + hashe[2] + "/" + hash_to_path2(hashe) + "/"  # choose to avoid overlapp
        else:
            #utc=burst.split(".")[2]
            event_name = burst.split(".",2)[2]
            #utc_date=datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%S.%f")

            hashe = hashtools.hashe_replace_object(hashe, burst, "any")

            # str("reduced hashe:",hashe,hash_to_path2(hashe))
            # if dataanalysis.printhook.global_log_enabled: print("reduced hashe:",hashe,hash_to_path2(hashe))
            #open("reduced_hashe.txt", "w").write(hash_to_path2(hashe) + "\n\n" + pprint.pformat(hashe) + "\n")
            #print(burst, hashe[2], burst)

            r = self.filecacheroot + "/byevent/"+event_name
            #r += "/%.4i" % utc_date.year
            #r += "/%.4i-%.2i" % (utc_date.year,utc_date.month)
            #r += "/%.4i-%.2i-%.2i" % (utc_date.year,utc_date.month,utc_date.day)
            #r += "/%s" % utc.replace(":","-")

            r += "/" + hashe[2] + "/" + hash_to_path2(hashe) + "/"

        # if dataanalysis.printhook.global_log_enabled: print("cached path:",r)

        print(self, "cached path:", r)

        return r  # choose to avoid overlapp

IntegralCacheRoots=os.environ.get('INTEGRAL_DDCACHE_ROOT',"./")

CacheStack=[]

for IntegralCacheRoot in IntegralCacheRoots.split(":"):
    ro_flag=False
    if IntegralCacheRoot.startswith("ro="):
        ro_flag=True
        IntegralCacheRoot=IntegralCacheRoot.replace("ro=","")

    mcgfb=CacheCounterparts(IntegralCacheRoot)
    mcgfb.readonly_cache=ro_flag
    if CacheStack==[]:
        CacheStack=[mcgfb]
    else:
        CacheStack[-1].parent=mcgfb
        CacheStack.append(mcgfb)

cache_counterpart=CacheStack[-1]

class DataAnalysis(da.DataAnalysis):
    store_preview_yaml=True

    cache=cache_counterpart

    def get_burst(self):
        if self._da_locally_complete is not None:
            try:
                return "(completeburst:%s)"%self.cache.get_scw(self._da_locally_complete)
            except:
                return "(complete)"

        for a in self.assumptions:
            if isinstance(a,Event):
                return "(burst:%s)"%str(a.t0_utc)

        return ""


class Event(DataAnalysis):
    event_name="unnamed"

    event_origin="unknown"

    healpix_url=""
                
    _nside = 256

    @property
    def nside(self):
     #   if not hasattr(self,'_nside'):
     #       lc = self.loc_map

        return self._nside
    

    @property
    def loc_region(self):
        indices = np.argsort(-self.loc_map)
        target_cum_map = np.empty(self.loc_map.shape)
        target_cum_map[indices] = 100 * np.cumsum(self.loc_map[indices])

        return target_cum_map

    def plot_map(self):
        healpy.mollview(self.loc_region,title="Localization of "+self.gname)
        healpy.graticule()
        for tr in np.linspace(0, 360 - 30, (360 / 30)):
            healpy.projtext(tr - 3, 3, "%i" % tr, lonlat=True)

    @property
    def gname(self):
        return self.event_name

    @property
    def trigger_time(self):
        return self.event_time['utc'] if hasattr(self, 'event_time') else 'unknown'

    def get_version(self):
        v = self.get_signature()+"."+self.version
        v += "." + self.event_name 
        v += "." + self.trigger_time.replace(":","")

        if len(self.healpix_url) > 0:
            s = self.healpix_url
            try:
                s=self.healpix_url.encode()
            except:
                pass

            v += ".loc-hp-"+hashlib.sha224(s).hexdigest()[:8]
        else:
            s = json.dumps(getattr(self, 'loc_parameters', {'loc':'unset'}))

            try:
                s = s.encode()
            except:
                pass

            v += ".loc-p-"+hashlib.sha224(s).hexdigest()[:8]

        return v

    def get_ra_dec(self):
        npx = np.arange(healpy.nside2npix(self.nside))
        return healpy.pix2ang(self.nside, npx, lonlat=True)

    @property
    def loc_map(self):
        if not hasattr(self,'_loc_map'):

            if len(self.healpix_url) > 0:
                # c = requests.get(self.healpix_url).content    

                # try:
                #     b = gzip.open(BytesIO(c))
                #     b.seek(0)
                #     f = fits.open(b)
                # except:
                #     f = fits.open(BytesIO(c))

                f = fits.open(self.healpix_url)

                self.map_header={}
                
                for k,v in dict(f[1].header).items():    
                    if k.startswith('LOG') or \
                       k.startswith('DIST') or \
                       k.startswith('INSTRUME'):
                        self.map_header[k] = v
                        print('map header',k,v)
                
                raw_loc_map = healpy.read_map(f[1])
                
                raw_nside = healpy.npix2nside(raw_loc_map.shape[0])

                
                peak_ra, peak_dec = healpy.pix2ang(raw_nside, raw_loc_map.argmax(), lonlat=True)
                self.loc_parameters = dict(ra=peak_ra, dec=peak_dec)
                
                print("healpix url NOT empty: peak at",self.loc_parameters)

                self._loc_map = healpy.get_interp_val(raw_loc_map, *self.get_ra_dec(), lonlat=True)
                self._loc_map/=np.sum(self._loc_map)
                
            else:
                print("healpix url empty: using point source location",self.loc_parameters)
                #self._loc_map=np.zeros(healpy.nside2npix(self.nside))

                ra,dec = self.get_ra_dec()

                p90_to_1sigma = -norm.isf(1 - (1 - .9) )


                dist_ra=(ra - self.loc_parameters['ra'])
                dist_ra[np.abs(dist_ra)>180]-=360
                    

                self._loc_map=np.exp(
                    -0.5*(dist_ra/(p90_to_1sigma*np.mean(np.abs(np.array(self.loc_parameters['dra_p90'])))))**2  \
                    -0.5*((dec - self.loc_parameters['dec']) / (p90_to_1sigma*np.mean(np.abs(np.array(self.loc_parameters['ddec_p90'])))))**2
                )

                self._loc_map/=np.sum(self._loc_map)

        return self._loc_map

    def peak(self):
        nside = healpy.npix2nside(self.loc_map.shape[0])
        peak_ra, peak_dec = healpy.pix2ang(nside, self.loc_map.argmax(), lonlat=True)
        return [float(peak_ra), float(peak_dec)]


    def describe_loc_regions(self):
        ra, dec= self.get_ra_dec()

        i_peak=self.loc_region.argmin()

        print("peak",ra[i_peak],dec[i_peak])

        for limit in 50, 90:
            m = self.loc_region < limit
            print(limit,"% containment")
            print("  RA",round(ra[i_peak]-ra[m].min(),2),'..',round(ra[m].max()-ra[i_peak],2),)
            print("  RA", round(ra[m].min(), 2), '..', round(ra[m].max(), 2), )
            print("  Dec", round(dec[i_peak] - dec[m].min(), 2), '..', round(dec[m].max() - dec[i_peak], 2), )
            print("  Dec", round(dec[m].min(), 2), '..', round(dec[m].max(), 2), )


    def main(self):
        self.setup()
        self.loc_map
        self.describe_loc_regions()

    def setup(self):
        pass

#class Event(da.DataAnalysis):
#    run_for_hashe=True

#    event_kind='LIGOEvent'

#    def main(self):
#        return da.get_object(self.event_kind)

class INTEGRALVisibility(DataAnalysis):
    input_target=Event

    minsolarangle=40
    follow_up_delay=0

    _da_settings=['minsolarangle','follow_up_delay']

    def main(self):
        try:
            ijd0=float(ic.converttime("UTC", self.input_target.trigger_time, "IJD"))
            ijd=ijd0+self.follow_up_delay
            utc= ic.converttime("IJD", "%.20lg"%ijd, "UTC")
        except Exception as e:
            utc=self.input_target.trigger_time

        self.visibility = integralvisibility.Visibility()
        self.visibility.minsolarangle = self.minsolarangle
        self.visibility_map = self.visibility.for_time(utc, nsides=self.input_target.nside)

    @property
    def total_visible(self):
        return np.sum(self.input_target.loc_map[self.visibility_map > 0])

    def peak_target_visible(self):
        ra,dec=grid_for_healpix_map(self.visibility_map)

        visible_map=np.copy(self.input_target.loc_map)
        visible_map[self.visibility_map<0.5]=0

        i=np.argmax(visible_map)
        return ra[i],dec[i]

    @property
    def nearest_to_peak_visible(self):
        return []

    def plot(self):
        healpy.mollview(self.visibility_map)
        healpy.graticule()

class INTEGRALVisibilitySummary(DataAnalysis):
    input_visibility=INTEGRALVisibility

    def main(self):
        self.peak_visible=self.input_visibility.peak_target_visible()
        self.total_visible=self.input_visibility.total_visible


class SourceAssumptions(DataAnalysis):

    version="v2"

    def main(self):
        data = {}
        data.update(dict(
            byinstrument={},
            instruments=["JEM-X", "ISGRI", "PICsIT", "SPI", "SPI-ACS", "IBIS/Veto"],
            zones=["fov", "z1", "z2", "z3"],
            typical_spectra=dict(
                hard=dict(model="compton", alpha=-0.5, epeak=600, beta=-9, ampl=1),
                soft=dict(model="band", alpha=-1, epeak=300, beta=-2.5, ampl=1),
                soft1s=dict(model="band", alpha=-1, epeak=300, beta=-2.5, ampl=1),
                crabby8s=dict(model="powerlaw", alpha=-2, epeak=300, beta=-2.5, ampl=1),
                crabby1s=dict(model="powerlaw", alpha=-2, epeak=300, beta=-2.5, ampl=1),
            )
        ))

        data['byinstrument']['JEM-X'] = dict(emin=3, emax=30, fov_area=130, angres="3'", zones=["fov", "onaxis"])
        data['byinstrument']['ISGRI'] = dict(emin=20, emax=200, fov_area=900, angres="12'", zones=["fov", "onaxis", "z1"])
        data['byinstrument']['PICsIT'] = dict(emin=260, emax=8000, fov_area=900, angres="30'",
                                              zones=["fov", "onaxis", "z1"])
        data['byinstrument']['SPI'] = dict(emin=25, emax=8000, fov_area=1225, angres="2.5$^\circ$", zones=["fov", "onaxis"])
        data['byinstrument']['SPI-ACS'] = dict(emin=75, emax=100000, fov_area=None, angres=None,
                                               zones=["fov", "onaxis", "z1", "z2", "z3"])
        data['byinstrument']['IBIS/Veto'] = dict(emin=100, emax=100000, fov_area=None, angres=None,
                                                 zones=["fov", "onaxis", "z1", "z2", "z3"])
        # data['instruments']['Compton']=dict(emin=260,emax=800)

        data['byinstrument']['JEM-X']['erange_ag'] = dict(emin=3, emax=30)
        data['byinstrument']['ISGRI']['erange_ag'] = dict(emin=20, emax=200)
        data['byinstrument']['PICsIT']['erange_ag'] = dict(emin=260, emax=2600)
        data['byinstrument']['SPI']['erange_ag'] = dict(emin=25, emax=1000)

        data['byinstrument']['SPI-ACS']['erange_grb'] = dict(emin=75, emax=2000)
        data['byinstrument']['IBIS/Veto']['erange_grb'] = dict(emin=100, emax=2000)
        data['byinstrument']['JEM-X']['erange_grb'] = dict(emin=3, emax=30)
        data['byinstrument']['ISGRI']['erange_grb'] = dict(emin=20, emax=200)
        data['byinstrument']['PICsIT']['erange_grb'] = dict(emin=260, emax=2600)
        data['byinstrument']['SPI']['erange_grb'] = dict(emin=25, emax=8000)

        self.data=data
        self.duration_by_kind = dict(hard=1, soft=8, soft1s=1, crabby1s=1, crabby8s=8)

class DataSource(DataAnalysis):
    # datasource='nrt'
    datasource='rt'

    def get_version(self):
        v=self.get_signature()+"."+self.version+"."+self.datasource
        return v
    

class InstrumentSelection(DataAnalysis):
    input_datasource=DataSource

    version="v1"

    def main(self):
        # if self.input_datasource.datasource == "nrt":
        #     self.targets = ["SPI-ACS", "IBIS/Veto", "ISGRI"]
        # elif self.input_datasource.datasource == "rt":
        self.targets = ["SPI-ACS"]

class OperationStatus(DataAnalysis):
    cached=True

    version="v2"

    input_ic=Event

    @property
    def targets(self):
        targets = []
        if self.ibis_on:
            targets += ["ISGRI", "IBIS/Veto"]
        if self.spi_on:
            targets += ["SPI-ACS", ]
        return targets

    def main(self):
        raise Exception("please override this for the given event")

class NoData(da.AnalysisException):
    pass

class DetectONExpected(DataAnalysis):
    input_target=Event
    force_online_expected = True
    #force_online_expected = False

    def main(self):
        if self.force_online_expected:
            print("FORCED online expected")
            return

        print("fetching realtime data for", self.input_target.trigger_time)
        self.rtdata_status = workflows.evaluate('odahub', 'integral-observation-summary', 'status', when_utc=self.input_target.trigger_time) # add as-of time
        print("got this",self.rtdata_status)


        if self.rtdata_status['data']['rtstatus'] != 'ONLINE': # this may change
            print("not ONLINE", self.rtdata_status['data']['rtstatus'])
            if self.rtdata_status['data']['rtstatus'] != 'Delayed': # this may change
                print("not Delayed", self.rtdata_status['data']['rtstatus'])
                print("no data!")
                raise NoData()

class DetectNoData(DataAnalysis):
    input_target=Event
    input_on=DetectONExpected

    input_datasource=DataSource


    def main(self):
        if self.input_datasource.datasource == "nrt":
            self.detect_nrt()

        if self.input_datasource.datasource == "rt":
            self.detect_rt()

    def detect_rt(self):
        pass
        
    def detect_nrt(self):
        sc = ic.get_sc(self.input_target.trigger_time)

        import time

        ntry = 120
        while ntry > 0:
            try:
                acs = ic.get_hk(target="ACS", utc = self.input_target.trigger_time,  span=300)
                break
            except Exception as e:

                try:
                    exception_response = e.args[0]
                    try:
                        exception_response = exception_response.decode()
                    except:
                        pass

                    if 'No Data (at this time?)' in exception_response and sc['bodies']['earth']['separation']<60e3:
                        raise NoData

                    print("unclassified exception",e,e.args)
                except NoData:
                    raise
                except Exception as e1:
                    print("exception in the exception",e, e1)

                print("trying", ntry)
                ntry -= 1
                time.sleep(3)

class OperationsReport(DataAnalysis):
    input_opsstatus=OperationStatus

    input_nodata = DetectNoData

    def main(self):

        if self.input_opsstatus.ibis_on and self.input_opsstatus.spi_on:
            self.text="INTEGRAL was operating in nominal mode"
        elif self.input_opsstatus.spi_on and not self.input_opsstatus.ibis_on:
            self.text="INTEGRAL SPI-ACS was operating in nominal mode, while IBIS was not taking data"
        elif self.input_opsstatus.ibis_on and not self.input_opsstatus.spi_on:
            self.text="INTEGRAL IBIS was operating in nominal mode, while SPI-ACS was not taking data"
        else:
            self.text = "INTEGRAL was not operational"


class CountLimits(DataAnalysis):
    input_target=Event
    input_assumptions=SourceAssumptions
    input_operation_status=OperationStatus
    input_nodata = DetectNoData
    input_iselect=InstrumentSelection
    input_datasource=DataSource

    cached = False

    span_s=300


    def get_version(self):
        v=self.get_signature()+"."+self.version
        if self.span_s!=600:
            v+="_span_%.5lg"%(self.span_s)
        return v

    version="v2.2"

    @property
    def targets(self):
        targets=[]
        if self.input_operation_status.ibis_on:
            targets+=["ISGRI", "IBIS/Veto"]
        if self.input_operation_status.spi_on:
            targets+=["SPI-ACS",]
        return [t for t in targets if t in self.input_iselect.targets]
    
    def get_count_limit(self,target, scale, nsig=None):
        if self.input_datasource.datasource == "rt":
            return self.get_count_limit_rt(target, scale, nsig)

        if self.input_datasource.datasource == "nrt":
            return self.get_count_limit_nrt(target, scale, nsig)

        # fail

    def get_ias(self, nrt=False):
        # ntries=50

        # if nrt:
        #     ep = {"rt": 0, "nrt": 1}
        # else:
        #     ep = {}

        # while ntries>0:
        #     try:
        #         ias_data = workflows.evaluate('odahub', 'integral-all-sky', 'integralallsky', t0_utc=self.input_target.trigger_time, **ep)
        #         if 'summary' in ias_data:
        #             print("managed to get useful rt data")
        #             break
        #     except Exception as e:
        #         print("failed to rt data: ", repr(e))

        #     print("waiting...",ntries)
        #     time.sleep(3)
        #     ntries-=1

        # self.ias_data = ias_data

        self.ias_data = json.load(open("integral_all_sky.json"))

        print("got ias data:")
        for k, v in self.ias_data.items():
            print(k, len(v), repr(v)[:300])

        self.excess_list = self.ias_data['reportable_excesses']
        self.excvar_summary = self.ias_data['excvar_summary']

    def get_count_limit_rt(self,target, scale, nsig=None):
        if target != 'ACS':
            raise Exception('unable to produce target %s, only ACS'%target)

        self.get_ias()

        if not hasattr(self,'hk'):
            self.hk={}

        if target not in self.hk:
            self.hk[target]={}

        _scale = scale

        # self.hk[target][scale] = self.ias_data['summary']['ACS_rt'][('s_%.3lg'%_scale).replace(".","_")]
        self.hk[target][scale] = self.ias_data['summary']['ACS'][('s_%.3lg'%_scale).replace(".","_")]

        self.hk[target][scale]['maxsig'] = self.hk[target][scale]['maxsnr']

        print(target, ":", scale, self.hk[target][scale])

        cl = self.hk[target][scale]['stdvar']*3*scale
        print("count_limit:", cl)


        return cl


    def get_durations_nrt(self):
        url = "http://lal.odahub.io/data/service/api/v2.0/integral-spiacs-duration/durations/api/v1.1/"+self.input_target.trigger_time+"/300"
        print("url",url)

        ntries_left = 20
        while ntries_left > 0:
            try:
                r = requests.get(url, auth=ic.get_auth())

                print("got this", r.text)

                self.duration_data = r.json()

                if self.duration_data.get('status') in [0., -2, -1]:
                    print("duration data ready", self.duration_data)
                    return self.duration_data

            except Exception as e:
                print("exception", repr(e))


            print("not ready", ntries_left)
            ntries_left -= 1
            time.sleep(5)

        raise Exception('duration data NOT ready')
        
            
    def get_count_limit_nrt(self,target, scale, nsig=None):
        n_attempts=50
        while n_attempts>=0:
            try:
                hk = ic.get_hk(
                                target=target,
                                utc=self.input_target.trigger_time,
                                span=self.span_s, t1=0, t2=0, ra=0, dec=0, rebin=scale,
                                vetofiltermargin=0.03
                            )['lc']
            except Exception as e:
            #except ic.ServiceException as e:
                print("waiting...",e)
                time.sleep(2)
                n_attempts-=1

            else:
                break
        if n_attempts<0:
            raise Exception("unable to reach the server in %i attempts!"%n_attempts)

        if not hasattr(self,'hk'):
            self.hk={}

        if target not in self.hk:
            self.hk[target]={}

        self.hk[target][scale]=hk
        
        self.get_ias(nrt=True)

        print(target, ":", scale, hk)
        return hk['count limit 3 sigma']

    def main(self):
        hkname = {'ISGRI': 'ISGRI', "PICsIT": "SPTI234", "IBIS/Veto": "IBIS_VETO", "SPI-ACS": "ACS"}

        count_limits = {}
        for kind, model in self.input_assumptions.data['typical_spectra'].items():
            if kind in self.input_assumptions.duration_by_kind:
                count_limits[kind] = {}
                for target in self.targets:
                    count_limits[kind][target] = self.get_count_limit(hkname[target], scale=self.input_assumptions.duration_by_kind[kind])

        if self.input_datasource.datasource == "nrt":
            self.duration_data = self.get_durations_nrt()
        else:
            self.duration_data = None

        self.count_limits=count_limits


class BackgroundStabilityAssertion(DataAnalysis):
    input_countlimits=CountLimits

    def main(self):
        excvar=[]
        instabilities=[]
        for target,t in self.input_countlimits.hk.items():
            for scale,d in t.items():
                print(d)

                excvar.append(d['excvar'])
                if d['maxsig']>6.:
                    instabilities.append(d['excvar'])

        if max(excvar)<1.2:
            self.text="very stable"
        elif max(excvar) < 1.5:
            self.text = "rather stable"
        elif max(excvar) < 2:
            self.text = "somewhat unstable"
        else:
            self.text = "very unstable"

        self.text+=" (excess variance %.2lg)"%max(excvar)

        if len(instabilities)>0:
            self.text+=" with some isolated features"

        if max(excvar)>2:
            self.text+=". We note that this amount of background variance may indicate source activity, even if it is not identified by the standard impulsive event detection pipeline"




class ScSystem(DataAnalysis):
    input_target=Event

    version="v2"

    def main(self):
        self.sc = ic.get_sc(self.input_target.trigger_time,
                            ra=self.input_target.loc_parameters['ra'],
                            dec=self.input_target.loc_parameters['dec'])

class Responses(DataAnalysis):
    input_assumptions=SourceAssumptions
    input_target=Event
    input_nodata = DetectNoData

    cached = False

    def main(self):
        CT = Counterpart()
        CT.nside = self.input_target.nside

        print("sides from target:", CT.nside)

        CT.utc = self.input_target.trigger_time
        CT.sc = ic.get_sc(CT.utc)
        CT.ra = CT.sc['scx']['ra']
        CT.dec = CT.sc['scx']['dec']
        CT.compute_transform_grids()

        responses = {}

        rname = {'ISGRI': 'ISGRI', "PICsIT": "PICsIT", "IBIS/Veto": "VETO", "SPI-ACS": "ACS"}
        lt_byt = dict(ISGRI=30, PICsIT=260, ACS="map2", VETO=100)

        for kind, model in self.input_assumptions.data['typical_spectra'].items():
            responses[kind] = {}
            print(model)

            for target in ["SPI-ACS", "ISGRI", "IBIS/Veto"]:

                n_attempts=10
                while True:
                    try:
                        mp = transform_rmap(np.array(ic.get_response_map(target=rname[target],
                                                                                  lt=lt_byt[rname[target]],
                                                                                  model=model['model'],
                                                                                  epeak=model['epeak'],
                                                                                  alpha=model['alpha'],
                                                                                  emin=75,
                                                                                  emax=2000,
                                                                                  )))
                    except Exception as e:
                        print("waiting",e)
                        time.sleep(1)
                        n_attempts-=1
                    else:
                        break

                skymp = CT.sc_map_in_sky(np.array(mp))
                responses[kind][target] = skymp

                if skymp is  None:
                    raise Exception("issue here")

        self.responses=responses

    def plot(self):
        for kind,v in self.responses.items():
            for instrument,mp in v.items():
                healpy.mollview(mp,title=kind+" "+instrument)


class OrientationComment(DataAnalysis):
    input_responses = Responses
    input_event = Event
    input_opsstatus = OperationStatus

    version="v2.2"

    def main(self):
        ra, dec = self.input_event.get_ra_dec()

        print("response, target sizes:", len(self.input_responses.responses['hard']['SPI-ACS']), len(ra))
        assert len(self.input_responses.responses['hard']['SPI-ACS'])==len(ra)

        i_peak = self.input_event.loc_region.argmin()
        self.response_onpeak = dict([
            [k, self.input_responses.responses['hard'][k][i_peak]] for k in self.input_opsstatus.targets
        ])

        self.response_best = dict([
            [k, self.input_responses.responses['hard'][k].min()] for k in self.input_opsstatus.targets
        ])

        self.response_quality={}

        for k in self.response_onpeak:
            if self.response_onpeak[k]/self.response_best[k]<1.5:
                self.response_quality[k]="near-optimal"
                print("->",self.response_quality[k])
            elif self.response_onpeak[k]/self.response_best[k]<2.5:
                self.response_quality[k]="somewhat suppressed"
                print("->",self.response_quality[k])
            else:
                self.response_quality[k]="strongly suppressed"
                print("->",self.response_quality[k])
    
            self.response_quality[k] += " (%.2lg%% of optimal)"%(100*(self.response_onpeak[k]/self.response_best[k])**-1)

            print("response of", k,"on-peak",self.response_onpeak[k], "best", self.response_best[k], "ratio", self.response_onpeak[k]/self.response_best[k], "quality", self.response_quality[k])


        print("peak", ra[i_peak], dec[i_peak], "response", self.response_onpeak)


        self.text="This orientation implies "

        k=list(self.response_onpeak.keys())[0]
        self.text+=self.response_quality[k]+" response of "+k

        for k in list(self.response_onpeak.keys())[1:-1]:
            self.text += ", " + self.response_quality[k] + " response of " + k

        k = list(self.response_onpeak.keys())[-1]
        self.text += ", and " + self.response_quality[k] + " response of " + k


        #near-optimal
        #    response of SPI-ACS, and this instrument provides best sensitivity to
        #    both short and long GRBs."""


class SensitivityMaps(DataAnalysis):
    input_assumptions=SourceAssumptions
    input_target=Event
    input_countlimits=CountLimits
    input_responses=Responses
    input_opsstatus=OperationStatus
    input_iselect=InstrumentSelection

    copy_cached_input=False

    cached = False

    def main(self):
        sens_maps = {}
        sens_maps_gal = {}

        theta_lvt, phi_lvt=grid_for_healpix_map(self.input_target.loc_map)

        for kind, model in self.input_assumptions.data['typical_spectra'].items():
            if kind in self.input_assumptions.duration_by_kind:
                sens_maps[kind] = {}
                sens_maps_gal[kind] = {}
                for target in self.input_opsstatus.targets:
                    if target not in self.input_iselect.targets:
                        continue

                    print("resp",kind,target,self.input_responses.responses[kind][target],"cl", self.input_countlimits.count_limits[kind][target])

                    print("self.input_responses.responses[kind][target]", self.input_responses.responses[kind][target])
                    print("self.input_countlimits.count_limits[kind][target]", self.input_countlimits.count_limits[kind][target])

                    r = self.input_responses.responses[kind][target] * self.input_countlimits.count_limits[kind][target]
                    #r[CT.sky_coord.separation(self.input_bodiesbody_coord).degree < bd["body_size"]] = 1e9

                    sens_maps[kind][target] = healpy.get_interp_val(r, theta_lvt, phi_lvt, lonlat=True)
                    sens_maps_gal[kind][target] = healpix_fk5_to_galactic(sens_maps[kind][target])

                sens_maps[kind]['best'] = np.array([m for k, m in sens_maps[kind].items()]).min(0)
                sens_maps_gal[kind]['best'] = np.array([m for k, m in sens_maps_gal[kind].items()]).min(0)

        self.sens_maps = sens_maps
        self.sens_maps_gal = sens_maps_gal

        self.plot_fancy()

        self.summarize()

    def plot(self):
        for kind, v in self.sens_maps.items():
            for instrument, mp in v.items():
                healpy.mollview(mp, title=kind + " " + instrument)



    def plot_fancy(self):
        p.figure()

        rangemin=dict(hard=1.5,soft=5)
        rangemax = dict(hard=6, soft=15)

        self.sens_map_plots={}

        for kind in ['hard','soft']:
            fig = healtics.plot_with_ticks(self.sens_maps[kind]['best'] / 1e-7,
                                           title="INTEGRAL 3-sigma upper limit on " + self.input_target.gname+ "; model "+kind,
                                       overplot=[[(self.input_target.loc_region, "k", 90), (self.input_target.loc_region, "k", 50),
                                                  #(mp, 'g', 0.5),
                                                  #   (fermi_shadow,'m',0.5),
                                                  # (region2,"r",90),(region2,"r",50),
                                                  # (sens_mcrab,"g",10),
                                                  #       (GC,"m",0.5),
                                                  ]],
                                       # overplot=[(region,"k",10),(healpy.smoothing(region2,0.2),"r",10)],
                                       # ilevels=array([50,10]),
                                       vmin=rangemin[kind], vmax=rangemax[kind], invra=False,
                                       unit="$10^{-7} \mathrm{erg^{}cm^{-2} s^{-1}}$")

            fn = "sens_map_%s_%s.png"%(kind,self.input_target.gname)
            self.sens_map_plots["sens_map_%s_png"%kind] = fn
            p.savefig(fn)

    def summarize(self):
        lvt = self.input_target.loc_region
        lvt_prob = self.input_target.loc_map

        self.summary={}

        for kind in "hard","soft", "soft1s", "crabby1s", "crabby8s":
            mp=self.sens_maps[kind]['best']

            acsresponse=self.sens_maps[kind]['SPI-ACS'] / self.input_countlimits.count_limits[kind]['SPI-ACS']

            self.summary[kind]=dict(
                p99_min=mp[lvt < 90].min(),
                p99_max=mp[lvt < 90].max(),
                p90_min=mp[lvt < 90].min(),
                p90_max=mp[lvt < 90].max(),
                p50_min=mp[lvt < 50].min(),
                p50_max=mp[lvt < 50].max(),
                p50_typical=np.quantile(mp[lvt < 50], 0.5),
                prob_within_1p5best=np.sum(lvt_prob[mp < 1.5 * mp[lvt < 90].min()]),

                p99_min_acsresponse=acsresponse[lvt < 90].min(),
                p99_max_acsresponse=acsresponse[lvt < 90].max(),
                p90_min_acsresponse=acsresponse[lvt < 90].min(),
                p90_max_acsresponse=acsresponse[lvt < 90].max(),
                p50_min_acsresponse=acsresponse[lvt < 50].min(),
                p50_max_acsresponse=acsresponse[lvt < 50].max(),
            )

            print("model", kind, "sensitivity in 90% containment extreme from",mp[lvt<90].min(),"to",mp[lvt<90].max())
            print("model", kind, "sensitivity in 50% containment extreme from", mp[lvt < 50].min(), "to",mp[lvt < 50].max())
            print("model", kind, "sensitivity within 50% of best contains",np.sum(lvt_prob[mp<1.5*mp[lvt<90].min()]))


class Sensitivities(DataAnalysis):
    input_sens = SensitivityMaps

    cached = True

    def main(self):
        self.summary = self.input_sens.summary
        for k,v in self.input_sens.sens_map_plots.items():
            setattr(self,k,v)

class DetectionSummary(DataAnalysis):
    input_countlimits=CountLimits
    input_datasource=DataSource
    input_sens=Sensitivities
    input_target=Event

    version="v1.1"
    
    @property
    def havedistance(self):
        return hasattr(self.input_target, 'map_header') and self.input_target.map_header.get('DISTMEAN') is not None


    @property
    def flux2lum(self):
        return float(self.input_target.map_header.get('DISTMEAN')*u.Mpc/u.cm)**2*4*np.pi

    def main(self):
        self.txt = ""

        self.excess_list = self.input_countlimits.excess_list

        for i_e in self.excess_list:
            if self.havedistance:
                i_e['excess']['luminosity_min'] = self.input_sens.summary['hard']['p50_min_acsresponse']*self.flux2lum*i_e['excess']['rate_overbkg']
                i_e['excess']['luminosity_max'] = self.input_sens.summary['hard']['p50_max_acsresponse']*self.flux2lum*i_e['excess']['rate_overbkg']
                i_e['excess']['luminosity'] = (i_e['excess']['luminosity_min'] + i_e['excess']['luminosity_max'])/2.
                i_e['excess']['luminosity_err_loc'] = (i_e['excess']['luminosity_max'] - i_e['excess']['luminosity_min'])/2. + i_e['excess']['luminosity_max']*0.2
                i_e['excess']['luminosity_err'] = self.input_sens.summary['hard']['p50_max_acsresponse']*self.flux2lum*i_e['excess']['rate_err']

                luminosities = [ i_e['excess'][l] for l in list(i_e['excess']) if l.startswith("luminosity") ]
                self.luminosity_scale = 10**min([int(np.log10(l)) for l in luminosities])

                for l in list(i_e['excess']):
                    if l.startswith("luminosity"):
                        i_e['excess'][l.replace("luminosity", "luminosity_scaled")] = i_e['excess'][l]/self.luminosity_scale
            else:
                i_e['excess']['flux_min'] = self.input_sens.summary['hard']['p50_min_acsresponse']*i_e['excess']['rate_overbkg']
                i_e['excess']['flux_max'] = self.input_sens.summary['hard']['p50_max_acsresponse']*i_e['excess']['rate_overbkg']
                i_e['excess']['flux'] = (i_e['excess']['flux_min'] + i_e['excess']['flux_max'])/2.
                i_e['excess']['flux_err_loc'] = (i_e['excess']['flux_max'] - i_e['excess']['flux_min'])/2. + i_e['excess']['flux_max']*0.2
                i_e['excess']['flux_err'] = self.input_sens.summary['hard']['p50_max_acsresponse']*i_e['excess']['rate_err']

                fluxes = [ i_e['excess'][l] for l in list(i_e['excess']) if l.startswith("flux") ]
                self.flux_scale = 10**min([int(np.log10(l)) for l in fluxes])

                for l in list(i_e['excess']):
                    if l.startswith("flux"):
                        i_e['excess'][l.replace("flux", "flux_scaled")] = i_e['excess'][l]/self.flux_scale

            i_e['excess']['FAP_sigma'] = stats.norm.isf(i_e['excess']['FAP'])


        self.excess_by_class = [
                        dict(comment="likely associated", fap_limit=0.003, excesses=[]), 
                        dict(comment="tentatively associated", fap_limit=0.01, excesses=[]), 
                        dict(comment="possibly associated", fap_limit=0.05, excesses=[]), 
                        dict(comment="likely background", fap_limit=1, excesses=[]), 
                    ]

        for excess in self.excess_list:
            for e_c in self.excess_by_class:
                if excess['excess']['FAP'] < e_c['fap_limit']:
                    e_c['excesses'].append(excess)
                    break
            
            #best_localfar_s = self.input_countlimits.rtdata['summary']['ACS_rt']['best']['localfar_s']
            #best_scale = self.input_countlimits.rtdata['summary']['ACS_rt']['best']['scale']
            #best_hk = self.input_countlimits.rtdata['summary']['ACS_rt'][('s_%.3lg'%best_scale).replace(".","_")]

            #bestsnr_hk = sorted(self.input_countlimits.rtdata['summary']['ACS_rt'].items(), key=lambda x:x[1].get('maxsnr',-999))[-1]

        best_fap = None
        best_t = None
        best_snr = None
        best_bkg_rate = None
        best_rate = None
        best_rate_err = None

        if self.input_datasource.datasource == 'nrt':
            duration_data = self.input_countlimits.duration_data

            if float(duration_data.get('status')) == 0.0:
                print("duration has detection", duration_data)

                if duration_data.get('spikeprob') == 0:
                    best_localfar_s = duration_data.get('maxsig')
                else:
                    best_localfar_s = stats.norm.isf(duration_data.get('spikeprob'))

                best_fap = best_localfar_s

                best_scale = 0.05 + [0, 2, 5, 11, 26, 61, 138, 316, 719, 1637, 3727, 8483, 19306, 43939, 100000][int(duration_data.get('maxsignalscale'))]*0.05*2
                

                best_t = duration_data.get('maxsig_time') - 300.
                best_snr = duration_data.get('maxsig')
                best_rate = duration_data.get('maxcounts') * 20.
                best_rate_err = duration_data.get('maxcounts_err') * 20.
                    
                self.detection_decision = True
                self.detection_associated = False
                self.detection_somewhat_associated = False
            else:
                print("duration did not yeild", duration_data)
                self.detection_decision = False
                self.detection_associated = False
                self.detection_somewhat_associated = False
                return
        else:

            for i_e in self.excess_list:
                if best_fap is None or i_e['excess']['FAP'] < best_fap:
                    print("new best FAP", i_e)
                    best_fap = i_e['excess']['FAP']
                    best_scale = i_e['scale']

                    best_t = i_e['excess']['rel_s_scale']
                    best_snr = i_e['excess']['snr']
                    best_localfar_s = best_snr
                    best_bkg_rate = i_e['excess']['rate'] - i_e['excess']['rate_overbkg']
                    best_rate = i_e['excess']['rate_overbkg']
                    best_rate_err = i_e['excess']['rate_err']
            

        self.excvar_summary = self.input_countlimits.excvar_summary

        best_fap_s = stats.norm.isf(best_fap)

        if best_fap_s < 2.5:
            self.detection_decision = False
            self.detection_associated = False
            self.detection_somewhat_associated = False

            return
        
        if best_fap_s > 3.:
            self.detection_decision = True
            self.detection_associated = True
            self.detection_somewhat_associated = True
        else:
            self.detection_decision = True
            self.detection_associated = False
            self.detection_somewhat_associated = True
        

        if best_snr < 5.:
            self.detection_decision = "marginal"
        elif best_snr < 7.:
            self.detection_decision = "moderately significant"
        else:
            self.detection_decision = "significant"

            
        txt = "We detect a %s event (S/N %.3lg) at %.3lgs time scale"%(self.detection_decision, best_snr, best_scale)
        if best_t is not None:
            txt += " at T0+%.5lg"%(best_t)

        txt+=". "

        txt += """Peak count rate of the signal in SPI-ACS is %.4lg cts/s, which corresponds to %.3lg to %.3lg erg/cm2/s depending on the location within the 50%% source localization region and assuming a typical short GRB spectrum. This estimate does not explore uncertainty related to the unknown event spectrum, systematic undertainty on the response of 20%%, or any dead-time correction. """%\
            (best_rate, 
             best_rate*self.input_sens.summary['hard']['p50_min_acsresponse'], 
             best_rate*self.input_sens.summary['hard']['p50_max_acsresponse'],
            )

        #if hasattr(self.input_target, 'map_header') and self.input_target.map_header.get('DISTMEAN') is not None:
        if self.havedistance:
            txt += "For the mean distance to the source of %.4lg Mpc this corresponds to the isotropic-equivalent luminosity between %.3lg to %.3lg erg/s. "%\
                (
                    self.input_target.map_header.get('DISTMEAN'),
                    best_rate*self.input_sens.summary['hard']['p50_min_acsresponse']*self.flux2lum,
                    best_rate*self.input_sens.summary['hard']['p50_max_acsresponse']*self.flux2lum,
                )
            

        if best_t is not None:
            if best_fap is not None:
                approx_FAP = best_fap
            else:
                approx_FAR_hz = 30./3600./24. 
                if best_snr>6:
                    approx_FAR_hz *= (best_snr/6.)**-2.7

                approx_FAP = 2 * approx_FAR_hz * abs(best_t) * (1+np.log( 30/0.1))

            if approx_FAP > 0.5:
                txt += "We derive preliminary estimate of the association FAP higher than %.2lg (less than %.2lg sigma). "%(0.5, stats.norm.isf(0.5))
            else:
                txt += "We derive preliminary estimate of the association FAP at the level of %.2lg (%.2lg sigma). "%(approx_FAP, stats.norm.isf(approx_FAP))

            if approx_FAP < 1e-2:
                txt += "This tentatively indicates a likely association. "
                self.detection_associated = True
            elif approx_FAP < 0.1:
                txt += "This tentatively indicates a random coincidence. "
                self.detection_associated = False
                self.detection_somewhat_associated = True
            else:
                txt += "This tentatively indicates a random coincidence. "
                self.detection_associated = False
                self.detection_somewhat_associated = False

        if best_rate is not None:
            self.maxrate = best_rate
            self.maxrate_scale_s = best_scale

        #if bestsnr_hk != best_hk:
        #    txt += "the highest S/N, on the other hand "
            

        txt += "Further analysis, taking into account accurate FAR measured on the basis of the study of the background during days surrounding the event will be reported in forthcoming circulars."

        self.txt = txt


class Counterpart(DataAnalysis):
    input_target=Event

    syst = 0.2

    t1 = 0
    t2 = 0

    do_burst_analysis = False

    def main(self):
        print(self.utc)

        self.sc = integralclient.get_sc(self.utc, ra=0, dec=0)

        if self.target_map_fn == "":
            self.target_map = np.zeros(healpy.nside2npix(16))
        else:
            self.target_map = healpy.read_map(self.target_map_fn)

        indices = np.argsort(-self.target_map)
        self.target_cum_map = np.empty(self.target_map.shape)
        self.target_cum_map[indices] = 100 * np.cumsum(self.target_map[indices])

        self.nside = healpy.npix2nside(self.target_map.shape[0])
        self.compute_transform_grids()

        if self.target_map_fn == "":
            cat = integralclient.get_cat(self.utc)
            locerr = cat['locerr']
            ra = cat['ra']
            dec = cat['dec']
            if locerr < 50:
                if locerr < 0.5:
                    locerr = 0.5
                self.target_map = np.array(
                    np.exp(-(self.sky_coord.separation(SkyCoord(ra, dec, unit=(u.deg, u.deg))) / u.deg) / locerr ** 2 / 2),
                    dtype=float)
                print(self.target_map.shape)

        if False:
            scales = [8]
            alpha = -1
            epeak = 300
            beta = -2.5
            model = "band"
            self.isgri_cl = 1.2
        else:
            scales = [1]
            alpha = -0.5
            epeak = 600
            beta = -2.5
            model = "compton"
            self.isgri_cl = 1.5

        self.response_mp_acs = transform_rmap(np.array(
            integralclient.get_response_map(target="ACS", lt='map2', alpha=alpha, epeak=epeak, beta=beta, model=model,
                                            kind="response")))
        self.response_mp_veto = transform_rmap(np.array(
            integralclient.get_response_map(target="VETO", lt=100, alpha=alpha, epeak=epeak, beta=beta, model=model,
                                            kind="response")))
        self.response_mp_isgri = transform_rmap(np.array(
            integralclient.get_response_map(target="ISGRI", lt=30, alpha=alpha, epeak=epeak, beta=beta, model=model,
                                            kind="response")))
        self.response_mp_picsit = transform_rmap(np.array(
            integralclient.get_response_map(target="PICsIT", lt=250, alpha=alpha, epeak=epeak, beta=beta, model=model,
                                            kind="response")))

        print("best response ACS, Veto, ISGRI", self.response_mp_acs.min(), self.response_mp_veto.min(), self.response_mp_isgri.min())

        def get_count_limit(target, scale):
            if target == 'ISGRI':
                return 1e6 # kill isgri!

            try:
                r = integralclient.get_hk(target=target, utc=self.utc, span=30.01, t1=0, t2=0, ra=0, dec=0, rebin=scale)
                hk = r.json()['lc']
            except Exception as e:
                print(e, r.content)

            print(target, ":", scale, hk)
            return hk['std bkg'] * 3 * hk['timebin']

        if self.do_burst_analysis:
            def get_burst_counts(target):
                span = (self.t2 - self.t1) * 2. + 100
                hk = integralclient.get_hk(target=target, utc=self.utc, span=span, t1=self.t1, t2=self.t2, ra=0, dec=0,
                                           rebin=0)['lc']
                print(target, ":", hk)
                return hk['burst counts'], hk['burst counts error'], hk['burst region']

            self.acs_counts = get_burst_counts("ACS")
            self.veto_counts = get_burst_counts("VETO")
            self.isgri_counts = get_burst_counts("ISGRI")
            self.picsit_counts = get_burst_counts("SPTI1234")

        # scales=[1,]
        # scales=[1,8,0.1]

        for scale in scales:
            acs_lim = get_count_limit("ACS", scale)
            veto_lim = get_count_limit("VETO", scale)
            isgri_lim = get_count_limit("ISGRI", scale)
            picsit_lim = isgri_lim * (3000 / 600) ** 0.5

            # acs_lim=300
            # veto_lim=300
            # isgri_lim=300


            print("ACS, Veto, ISGRI", acs_lim, veto_lim, isgri_lim)

            self.syst = 0.

            sens_map = self.response_mp_acs * acs_lim * (self.syst + 1)
            sens_map_acs = self.response_mp_acs * acs_lim * (self.syst + 1)
            sens_map_veto = self.response_mp_veto * veto_lim * (self.syst + 1)
            sens_map_isgri = self.response_mp_isgri * isgri_lim * (self.syst + 1) * self.isgri_cl
            sens_map_picsit = self.response_mp_picsit * picsit_lim * (self.syst + 1)
            sens_map[sens_map > sens_map_veto] = sens_map_veto[sens_map > sens_map_veto]
            sens_map[sens_map > sens_map_isgri] = sens_map_isgri[sens_map > sens_map_isgri]
            sens_map[sens_map > sens_map_picsit] = sens_map_picsit[sens_map > sens_map_picsit]

            na = sens_map[~np.isnan(sens_map) & (sens_map > 0)].min()

            na_e = int(np.log10(na)) - 1
            na_b = int(na * 10 / 10 ** na_e) / 10.

            ### 

            # na_b=3
            # na_e=-7
            na_b = 1.3

            ###

            na = na_b * 10 ** na_e

            print("best ACS", na)
            nv = sens_map_veto[~np.isnan(sens_map_veto) & (sens_map_veto > 0)].min()
            print("best VETO", nv)
            self.sens_scale = na_b * 10 ** na_e
            sens_map /= self.sens_scale
            sens_map_veto /= self.sens_scale
            sens_map_acs /= self.sens_scale
            sens_map_isgri /= self.sens_scale
            sens_map_picsit /= self.sens_scale

            self.sens_scale_e = na_e
            self.sens_scale_b = na_b

            self.sens_map = sens_map
            self.sens_map_acs = sens_map_acs
            self.sens_map_veto = sens_map_veto
            self.sens_map_isgri = sens_map_isgri
            self.sens_map_picsit = sens_map_picsit

            self.tag = "_%.5lgs" % scale

            if self.do_burst_analysis:
                self.localize()
                self.localize2()
                self.localization()
            self.compute_maps()

    def localize(self):

        allmaps = []
        alldet = [[(c, ce, br), m, n] for (c, ce, br), m, n in [(self.acs_counts, self.response_mp_acs, "ACS"),
                                                                (self.veto_counts, self.response_mp_veto, "VETO"),
                                                                (self.isgri_counts, self.response_mp_isgri, "ISGRI"),
                                                                (self.picsit_counts, self.response_mp_picsit, "PICsIT")]
                  if c / ce > -3 and
                  ce > 0 and
                  br > 0 and
                  c != 0
                  ]

        print("will localize with", [n for (c, ce, br), m, n in alldet])

        total_region = []
        c_vec = np.array([c for (c, ce, br), m, n in alldet])
        ce_vec = np.array([ce for (c, ce, br), m, n in alldet])
        # ce_vec=(ce_vec**2+(c_vec*0.05)**2)**0.5

        response_mt = np.array([m for ((c, ce, br), m, n) in alldet])
        print(response_mt.shape)

        nc_mt = np.outer(c_vec, np.ones(response_mt.shape[1])) * response_mt
        nce_mt = np.outer(ce_vec, np.ones(response_mt.shape[1])) * response_mt

        mean_map = np.sum(nc_mt / nce_mt ** 2, axis=0) / sum(1. / nce_mt ** 2, axis=0)
        err_map = 1 / np.sum(1. / nce_mt ** 2, axis=0) ** 0.5
        chi2_map = np.sum((nc_mt - np.outer(np.ones_like(c_vec), mean_map)) ** 2 / nce_mt ** 2, axis=0)

        min_px = chi2_map.argmin()
        print("minimum prediction", response_mt[:, min_px], mean_map[min_px] / response_mt[:, min_px], chi2_map[min_px])
        print("measure", c_vec)
        print("measure err", ce_vec)
        print("sig", (c - mean_map[min_px] * response_mt[:, min_px]) / ce_vec)

        healpy.mollview(chi2_map)
        plot.plot("chi2_map.png")

        self.locmap = chi2_map / chi2_map.min()

        print(self.locmap.min(), self.locmap.max())

        healpy.mollview(mean_map)
        plot.plot("mean_map.png")

    def localize2(self):

        allmaps = []
        alldet = [(self.acs_counts, self.response_mp_acs, "ACS"),
                  (self.veto_counts, self.response_mp_veto, "VETO"),
                  (self.picsit_counts, self.response_mp_picsit, "PICsIT")]

        #         (self.isgri_counts,self.response_mp_isgri,"ISGRI"),

        total_region = np.ones_like(self.response_mp_acs, dtype=bool)
        for i1, ((c1, ce1, br1), m1, n1) in enumerate(alldet):
            for i2, ((c2, ce2, br2), m2, n2) in enumerate(alldet):
                if c1 == 0 or c2 == 0: continue  # ???
                if ce1 == 0 or ce2 == 0: continue
                if br1 == 0 or br2 == 0: continue
                if i2 >= i1: continue

                ang0 = np.arctan2(m2, m1)  # inversed for responose
                h0 = np.histogram(ang0.flatten(), 100)

                ang1 = np.arctan2((c1 - ce1) * (1 - self.syst), (c2 + ce2) * (1 + self.syst))
                ang2 = np.arctan2((c1 + ce1) * (1 + self.syst), (c2 - ce2) * (1 - self.syst))

                print(n1, ":", c1, ce1, "; ", n2, c2, ce2, " => ", ang1, ang2, " while ", ang0.min(), ang0.max())

                region = (ang0 > ang1) & (ang0 < ang2)

                healpy.mollview(region)
                plot.plot("region_%s_%s.png" % (n1, n2))

                total_region &= region

        healpy.mollview(total_region)
        plot.plot("total_region.png")

    def localization(self):
        syst = 0.02

        fluence = self.sens_map_acs.min() * 10

        print("fluence for 10 sigma in ACS:", fluence)

        sig_map_acs = fluence / self.sens_map_acs
        sig_map_veto = fluence / self.sens_map_veto
        sig_map_isgri = fluence / self.sens_map_isgri
        sig_map_picsit = fluence / self.sens_map_picsit

        if False:
            figure()
            scatter(sig_map_acs, sig_map_picsit, c=coord.theta)
            colorbar()

            figure()
            scatter(sig_map_acs, sig_map_veto, c=coord.theta)
            colorbar()

            figure()
            scatter(sig_map_acs, sig_map_isgri, c=coord.theta)
            colorbar()

            figure()
            scatter(sig_map_veto, sig_map_picsit, c=coord.theta)
            colorbar()

        allmaps = []
        allsig = [sig_map_acs,
                  sig_map_veto,
                  sig_map_isgri,
                  sig_map_picsit]

        for m1 in allsig:
            for m2 in allsig:
                # m1=m1/sig_map_acs*sig_map_acs.max()
                # m2=m1/sig_map_acs*sig_map_acs.max()

                ang0 = np.arctan2(m1, m2)
                h0 = np.histogram(ang0.flatten(), 100)
                ang1 = np.arctan2((m1 - 1) * (1 - syst), (m2 + 1) * (1 + syst))
                ang2 = np.arctan2((m1 + 1) * (1 + syst), (m2 - 1) * (1 - syst))

                areas = [np.sum((ang0[i] > ang1) & (ang0[i] < ang2)) for i in range(ang0.shape[0])]

                area = (np.array(areas) * healpy.nside2pixarea(16))
                area /= 4 * np.pi

                allmaps.append(area)

        bestarea = np.array(allmaps).min(0)
        healpy.mollview(bestarea)
        self.bestarea = bestarea
        plot.plot("bestlocalization.png")

        plot.p.figure()
        plot.p.a = plot.p.hist(bestarea, np.logspace(-5, -0.5, 100), log=True)
        plot.p.semilogx()
        plot.plot("bestlocalization_hist.png")

    def get_grid(self, nside=None):
        nside = nside if nside is not None else self.nside
        npx = np.arange(healpy.nside2npix(nside))
        theta, phi = healpy.pix2ang(nside, npx)
        return SkyCoord(phi, theta, 1, unit=(u.rad, u.rad), representation_type="physicsspherical")
        # return SkyCoord(phi, theta, 1, unit=(u.rad, u.rad))

    def compute_transform_grids(self):
        sky_coord = self.get_grid()
        self.sky_coord = sky_coord

        x = sky_coord.cartesian.x
        y = sky_coord.cartesian.y
        z = sky_coord.cartesian.z

        self.scX = SkyCoord(self.sc['scx']['ra'], self.sc['scx']['dec'], frame='icrs', unit='deg')
        self.scY = SkyCoord(self.sc['scy']['ra'], self.sc['scy']['dec'], frame='icrs', unit='deg')
        self.scZ = SkyCoord(self.sc['scz']['ra'], self.sc['scz']['dec'], frame='icrs', unit='deg')
        self.scz = self.scX
        self.scy = self.scY
        self.scx = self.scZ

        # maps of sky coordinates in detector coordinate grid
        x_sky_in_sc = sky_coord.cartesian.x * self.scx.cartesian.x + sky_coord.cartesian.y * self.scy.cartesian.x + sky_coord.cartesian.z * self.scz.cartesian.x
        y_sky_in_sc = sky_coord.cartesian.x * self.scx.cartesian.y + sky_coord.cartesian.y * self.scy.cartesian.y + sky_coord.cartesian.z * self.scz.cartesian.y
        z_sky_in_sc = sky_coord.cartesian.x * self.scx.cartesian.z + sky_coord.cartesian.y * self.scy.cartesian.z + sky_coord.cartesian.z * self.scz.cartesian.z

        sky_in_sc = SkyCoord(x=x_sky_in_sc, y=y_sky_in_sc, z=z_sky_in_sc, representation_type="cartesian")

        on_scz = self.scz.cartesian.x * sky_coord.cartesian.x + self.scz.cartesian.y * sky_coord.cartesian.y + self.scz.cartesian.z * sky_coord.cartesian.z
        x_inxy = x - on_scz * self.scz.cartesian.x
        y_inxy = y - on_scz * self.scz.cartesian.y
        z_inxy = z - on_scz * self.scz.cartesian.z

        r_inxy = (x_inxy * x_inxy + y_inxy * y_inxy + z_inxy * z_inxy) ** 0.5
        x_inxy /= r_inxy
        y_inxy /= r_inxy
        z_inxy /= r_inxy

        sc_in_sky = SkyCoord(
            np.arctan2(x_inxy * self.scy.cartesian.x + y_inxy * self.scy.cartesian.y + z_inxy * self.scy.cartesian.z,
                    x_inxy * self.scx.cartesian.x + y_inxy * self.scx.cartesian.y + z_inxy * self.scx.cartesian.z),
            np.arccos(on_scz),
            1,
            unit=(u.rad, u.rad),
            representation_type="physicsspherical")

        self.sky_in_sc = sky_in_sc
        self.sc_in_sky = sc_in_sky

        return sky_in_sc, sc_in_sky

    def sky_map_in_sc(self, sky_map):
        return healpy.get_interp_val(sky_map,
                                     theta=self.sky_in_sc.represent_as("physicsspherical").theta.rad,
                                     phi=self.sky_in_sc.represent_as("physicsspherical").phi.rad)

    def sc_map_in_sky(self, sc_map):
        return healpy.get_interp_val(sc_map,
                                     theta=self.sc_in_sky.represent_as("physicsspherical").theta.rad,
                                     phi=self.sc_in_sky.represent_as("physicsspherical").phi.rad)

    def plot_sky_diagram(self):
        target_map_in_sc = self.sky_map_in_sc(self.target_map)
        target_map_in_sc[
            np.abs(self.sky_coord.represent_as("physicsspherical").theta.deg) < 20] += target_map_in_sc.max() / 5.
        target_map_in_sc[
            np.abs(self.sky_coord.represent_as("physicsspherical").theta.deg) > 140] += target_map_in_sc.max() / 5.
        target_map_in_sc[(self.sky_coord.represent_as("physicsspherical").phi.deg < 30) | (
            self.sky_coord.represent_as("physicsspherical").phi.deg > 360 - 30)] += target_map_in_sc.max() / 10.
        healpy.mollview(target_map_in_sc, cmap='YlOrBr')
        healpy.projscatter(np.pi / 2, 0)
        healpy.graticule()
        plot.plot()

    def compute_maps(self):
        healpy.mollview(self.sens_map_acs, cmap="YlOrBr")
        healpy.graticule()

        plot.plot()

        sens_map_sky = healpy.sphtfunc.smoothing(self.sc_map_in_sky(self.sens_map), 5. / 180. * np.pi)
        sens_map_sky_acs = healpy.sphtfunc.smoothing(self.sc_map_in_sky(self.sens_map_acs), 5. / 180. * np.pi)
        sens_map_sky_veto = healpy.sphtfunc.smoothing(self.sc_map_in_sky(self.sens_map_veto), 5. / 180. * np.pi)
        sens_map_sky_isgri = healpy.sphtfunc.smoothing(self.sc_map_in_sky(self.sens_map_isgri), 5. / 180. * np.pi)
        sens_map_sky_picsit = healpy.sphtfunc.smoothing(self.sc_map_in_sky(self.sens_map_picsit), 5. / 180. * np.pi)

        if self.do_burst_analysis:
            bestarea_sky = healpy.sphtfunc.smoothing(self.sc_map_in_sky(self.bestarea), 5. / 180. * np.pi)
            locmap_sky = healpy.sphtfunc.smoothing(self.sc_map_in_sky(self.locmap), 5. / 180. * np.pi)

        good_mask = lambda x: sens_map_sky < sens_map_sky.min() * x
        print("good for", [np.sum(self.target_map[good_mask(x)]) for x in [1.01, 1.1, 1.2, 1.5, 2.]])

        print("very good", sens_map_sky.min())
        print("typical bad", np.sum(self.target_map[~good_mask(1.2)] * sens_map_sky[~good_mask(1.2)]) / np.sum(),
            self.target_map[~good_mask(1.2)])
        print("typical good", np.sum(self.target_map[good_mask(1.2)] * sens_map_sky[good_mask(1.2)]) / np.sum(),
            self.target_map[good_mask(1.2)])

        # map_sc=healpy.sphtfunc.smoothing(map_sc,5./180.*pi)
        # target_cum_map_sm=healpy.sphtfunc.smoothing(self.target_cum_map,2./180.*pi)

        # overplot=[]
        try:
            o_isgrimap = np.loadtxt(gzip.open("isgri_sens.txt.gz"))
            o_isgrimap[np.isnan(o_isgrimap) | np.isinf(o_isgrimap)] = 0

            isgrimap = healpy.get_interp_val(o_isgrimap,
                                             self.sky_coord.represent_as("physicsspherical").theta.rad,
                                             self.sky_coord.represent_as("physicsspherical").phi.rad,
                                             )

            o_jemxmap = np.loadtxt(gzip.open("jemx_sens.txt.gz"))
            o_jemxmap[np.isnan(o_jemxmap) | np.isinf(o_jemxmap)] = 0
            jemxmap = healpy.get_interp_val(o_jemxmap,
                                            self.sky_coord.represent_as("physicsspherical").theta.rad,
                                            self.sky_coord.represent_as("physicsspherical").phi.rad,
                                            )

            o_spimap = np.loadtxt(gzip.open("spi_sens.txt.gz"))
            o_spimap[np.isnan(o_spimap) | np.isinf(o_spimap)] = 0
            spimap = healpy.get_interp_val(o_spimap,
                                           self.sky_coord.represent_as("physicsspherical").theta.rad,
                                           self.sky_coord.represent_as("physicsspherical").phi.rad,
                                           )

            cover_info = {}

            for detname, detmap, o_detmap in [("isgri", isgrimap, o_isgrimap),
                                              ("jemx", jemxmap, o_jemxmap),
                                              ("spi", spimap ** 2, o_spimap ** 2)]:
                bestsens = min(o_detmap[o_detmap > 5.9e-6 ** 2])
                print("min", bestsens, bestsens ** 0.5)
                cover = (detmap < bestsens * 20 ** 2) & (detmap > 0)
                print("contained in", detname, "area", np.sum(cover) / float(cover.shape[0]) * 4 * np.pi * (180 / np.pi) ** 2, self.target_map.sum(), \
                    self.target_map[cover].sum(), np.sum(cover) * 1. / cover.shape[0], np.sum(cover) * 1. / cover.shape[
                    0] * 4 * np.pi * (180 / np.pi) ** 2)

                cover_info[detname] = dict(
                    area_deg2=np.sum(cover) / float(cover.shape[0]) * 4 * np.pi * (180 / np.pi) ** 2,
                    target_coverage=self.target_map[cover].sum()
                )

            json.dump(cover_info, open("coverage_info.json", "w"))

            overplot = [(self.target_cum_map, "k", [50, 90]),
                        (spimap, "g", spimap[spimap > 0].min() * 20),
                        (jemxmap, "b", jemxmap[jemxmap > 0].min() * 20 ** 2),
                        (isgrimap, "r", isgrimap[isgrimap > 0].min() * 20 ** 2)],
        except Exception as e:
            raise
            if np.sum(target_cum_map_sm > 0) > 0:
                overplot = [(target_map_sm, "k", None)]
                # overplot.append((locmap_sky, "gist_gray", None))



                # cover=theta_sc_rad>120./180*pi
                # print("contained in >120",map_px[cover].sum(),sum(cover)*1./cover.shape[0],sum(cover)*1./cover.shape[0]*4*pi*(180/pi)**2)

                # cover=(theta_sc_rad>80./180*pi) & (theta_sc_rad<120./180*pi)
                # print("contained in 80-120",map_px[cover].sum(),sum(cover)*1./cover.shape[0],sum(cover)*1./cover.shape[0]*4*pi*(180/pi)**2)

                # cover=(theta_sc_rad<80./180*pi)
                # print("contained in <80",map_px[cover].sum(),sum(cover)*1./cover.shape[0],sum(cover)*1./cover.shape[0]*4*pi*(180/pi)**2)
                #       sens_mp_sky[]*=2

        for body_name in "earth", "moon", "sun":
            bd = self.sc['bodies'][body_name]
            body_coord_sc = SkyCoord(bd['body_in_sc'][1], bd['body_in_sc'][0], 1, unit=(u.deg, u.deg),
                                     representation_type="physicsspherical")
            body_coord = SkyCoord(bd['body_ra'], bd['body_dec'], unit=(u.deg, u.deg))
            print("body:", body_name, bd)
            print("body coordinates:", bd['body_ra'], bd['body_dec'], body_coord)
            sens_map_sky[self.sky_coord.separation(body_coord).degree < bd["body_size"]] = 1e9

        healpy.write_map("sens_map_sky_" + self.tag + ".fits", sens_map_sky)

        if self.do_burst_analysis:
            p = healtics.plot_with_ticks(locmap_sky, cmap="jet", title="",
                                         unit="",
                                         vmin=1, vmax=9,
                                         overplot=overplot)
            plot.plot("sky_locmap.png", format='png', dpi=100)

            p = healtics.plot_with_ticks(locmap_sky, cmap="jet", title="",
                                         unit="",
                                         vmin=1, vmax=locmap_sky.max(),
                                         overplot=overplot)
            plot.plot("sky_locmap_full.png", format='png', dpi=100)

        def saveplot(prefix):
            plot.plot(prefix + self.tag + ".svg", format='svg', dpi=200)

        p = healtics.plot_with_ticks(sens_map_sky * self.sens_scale_b, cmap="YlOrBr", title="",
                                     unit="$10^{%i} \mathrm{erg^{ }cm^{-2}}$" % self.sens_scale_e,
                                     overplot=overplot,
                                     vmin=self.sens_scale_b, vmax=10 * self.sens_scale_b)
        saveplot("sky_sens_")

        p = healtics.plot_with_ticks(sens_map_sky_acs * self.sens_scale_b, cmap="YlOrBr", title="",
                                     overplot=overplot,
                                     unit="$10^{%i} \mathrm{erg^{ }cm^{-2}}$" % self.sens_scale_e,
                                     vmin=self.sens_scale_b, vmax=10 * self.sens_scale_b)
        saveplot("sky_sens_acs_")

        p = healtics.plot_with_ticks(sens_map_sky_veto * self.sens_scale_b, cmap="YlOrBr", title="",
                                     overplot=overplot,
                                     unit="$10^{%i} \mathrm{erg^{ }cm^{-2}}$" % self.sens_scale_e,
                                     vmin=self.sens_scale_b, vmax=10 * self.sens_scale_b)
        saveplot("sky_sens_veto_")

        p = healtics.plot_with_ticks(sens_map_sky_isgri * self.sens_scale_b, cmap="YlOrBr", title="",
                                     overplot=overplot,
                                     unit="$10^{%i} \mathrm{erg^{ }cm^{-2}}$" % self.sens_scale_e,
                                     vmin=self.sens_scale_b, vmax=10 * self.sens_scale_b)
        saveplot("sky_sens_isgri_")

        p = healtics.plot_with_ticks(sens_map_sky_picsit * self.sens_scale_b, cmap="YlOrBr", title="",
                                     overplot=overplot,
                                     unit="$10^{%i} \mathrm{erg^{ }cm^{-2}}$" % self.sens_scale_e,
                                     vmin=self.sens_scale_b, vmax=10 * self.sens_scale_b)
        saveplot("sky_sens_picsit_")

        if self.do_burst_analysis:
            p = healtics.plot_with_ticks(bestarea_sky * 100, cmap="YlOrBr", title="",
                                         overplot=overplot,
                                         unit="% of the sky",
                                         vmin=1, vmax=100)
            plot.plot("sky_sens_locarea_" + self.tag + ".png", format='png', dpi=100)

            # p=healtics.plot_with_ticks(sens_mp_sky,cmap="jet",title="INTEGRAL SPI-ACS 3 sigma upper limit in 1 second",overplot=[(map_px,"gist_gray",None),(spimap,"summer",20),(jemxmap,"winter",20**2),(isgrimap,"autumn",20**2)],vmin=0,vmax=sens_mp_sky.max())
            # p=healtics.plot_with_ticks(sens_mp_sky,cmap="YlOrBr",title="INTEGRAL SPI-ACS 3 sigma upper limit in 1 second",overplot=[(map_px,"gist_gray",None),(spimap,"summer",20),(jemxmap,"winter",20**2),(isgrimap,"autumn",20**2)],vmin=0,vmax=sens_mp_sky.max())
            # p=healtics.plot_with_ticks(sens_mp_sky,cmap="YlOrBr",title="INTEGRAL SPI-ACS 3 sigma upper limit in 1 second",overplot=healpy.sphtfunc.smoothing(map_px,5./180.*pi),vmin=1.5,vmax=15)

            # plot.plot("sky_sens.svg", format='svg', dpi=200)

    def plot_raw_sky(self):
        healpy.mollview(self.target_map, cmap='YlOrBr')
        healpy.projtext(self.scx.represent_as("physicsspherical").theta.rad,
                        self.scx.represent_as("physicsspherical").phi.rad,
                        "scx: SCZ")
        healpy.projscatter(self.scx.represent_as("physicsspherical").theta.rad,
                           self.scx.represent_as("physicsspherical").phi.rad)
        healpy.projtext(self.scy.represent_as("physicsspherical").theta.rad,
                        self.scy.represent_as("physicsspherical").phi.rad,
                        "scy: SCY")
        healpy.projscatter(self.scy.represent_as("physicsspherical").theta.rad,
                           self.scy.represent_as("physicsspherical").phi.rad)

        healpy.projtext(self.scz.represent_as("physicsspherical").theta.rad,
                        self.scz.represent_as("physicsspherical").phi.rad,
                        "scz: SCX")
        healpy.projscatter(self.scz.represent_as("physicsspherical").theta.rad,
                           self.scz.represent_as("physicsspherical").phi.rad)
        healpy.graticule()

        plot.plot("rawsky.png")

class FinalComment(DataAnalysis):
    input_ic=Event

    cached=True

    def main(self):
        raise Exception("please redefine")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='make counterpart')
    parser.add_argument('--target-map', dest='targetmap', action='store', default="",
                        help='map of the target')
    parser.add_argument('--target-position', dest='targetposition', action='store', default="",
                        help='location of the target')
    parser.add_argument('--utc', dest='utc', action='store',
                        help='utc')
    parser.add_argument('--t1', dest='t1', action='store', default="0",
                        help='t1')
    parser.add_argument('--t2', dest='t2', action='store', default="0",
                        help='t2')

    args = parser.parse_args()

    Counterpart(use_target_map_fn=args.targetmap, use_utc=args.utc, use_t1=float(args.t1), use_t2=float(args.t2)).get()

