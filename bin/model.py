
# Basic model utilities

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.spatial import cKDTree
import pickle
import collections

from .absence import data as absence_data

import pandas as pd
import numpy as np
from .loader import get_predictor_data, array_to_raster
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage, cut_tree
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from geopy.distance import geodesic
import gdal

class TweakedPipeline(Pipeline):

    def __getattr__(self, name):
        if name in ['coef_', 'feature_importances_']:
            if hasattr(self.steps[-1][-1], name):
                return getattr(self.steps[-1][-1], name, None)
        raise AttributeError


class PreprocessingMixin(BaseEstimator, TransformerMixin):

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df


class SelectDataWithinArea(PreprocessingMixin):
    '''
    All outside the bounding box data is filtered out.

    :param

        bbox -- [latmin, lonmin, latmax, lonmax], i.e it is an array like object.
    '''
    def __init__(self, bbox=None):
        self.bbox_ = bbox

    def transform(self, df, y=None):
        if self.bbox_ is None: return df
        latmin, lonmin, latmax, lonmax = tuple(self.bbox_)
        df_ = df.copy()
        inds = (df_.latitude <= latmax) * (df_.latitude >= latmin) * \
        (df_.longitude <= lonmax) * (df_.longitude >= lonmin)
        return df_[inds].reset_index(drop=True)


class SelectDataWithinPolygon(PreprocessingMixin):
    '''
    All outside the bounding box data is filtered out.
    '''

    def transform(self, df, y=None):
        df_ = df.copy()
        condition = (((df_.longitude < 92) & (df_.latitude > 62))
                     | ((df_.longitude< 113) & (df_.latitude > 57) & (df_.latitude < 62))
                     | ((df_.latitude < 57) & (df_.longitude < 110))
        )
        return df_[~condition].reset_index(drop=True)



class DensityTweaker(PreprocessingMixin):

    def __init__(self, density=0.1):
        self.density_ = density

    def transform(self, df, y=None):
        df_ = df.copy()
        to_remove = []
        for j in range(df_.shape[0]):
            lost_df = df_.drop(to_remove)
            lats = lost_df.latitude.values
            lons = lost_df.longitude.values
            ixes = lost_df.index.values
            clat = df_.iloc[j].latitude
            clon = df_.iloc[j].longitude
            inds = (lats >= (clat - 0.5)) * (lats <= (clat + 0.5)) * \
                   (lons >= (clon - 0.5)) * (lons <= (clon + 0.5))
            if inds.sum() > self.density_:
                print("Current point:", clat, clon, 'N=', inds.sum())
                num_to_remove = int(inds.sum() - self.density_)
                if num_to_remove:
                    choiced = np.random.choice(ixes[inds],
                                               num_to_remove,
                                               replace=False)
                    to_remove += list(choiced)

        if len(to_remove) == 0:
            print("Nothing to remove; points density is good.")
        else:
            print("REMOVED POINTS: for %s"  % df_.species[0])
            print("Points were removed, total: %s" % len(to_remove))
        to_remove = list(set(to_remove))

        return df_.drop(to_remove).reset_index(drop=True)


class DensityThinner(PreprocessingMixin):

    def __init__(self, distance=1):
        """distance is expected to be provided in km"""
        self.distance = distance


    def transform(self, df, y=None):
        df_ = df.copy()

        lats = df_.latitude.values
        lons = df_.longitude.values

        X = np.vstack([lats, lons]).T * np.pi / 180.0

        ckdtree = cKDTree(X)
        to_remove = np.array([False] * len(lats))
        for ind in range(len(to_remove)):
            if to_remove[ind] is False:
                results = ckdtree.query_ball_point(X[ind, :], self.distance)
                to_remove[[_ for _ in results if _ != ind]] = True

        res = df_.drop(df_.index.values[to_remove]).reset_index(drop=True)
        res.to_csv("Larix_pruned.csv")
        sdfkl
        return  res


class FillPresenceAbsenceFromFile(PreprocessingMixin):

    def __init__(
        self,
        filename='./data/larix_113_123.tif',
        presence_values=(113, 123),
        num_to_select=10000,
        nan_value=200,
        species='larix'
    ):
        self.filename = filename
        self.presence_values = presence_values
        self.nan_value = nan_value
        self.num_to_select = num_to_select
        self.species = species

    def convert_inds_to_coords(self, lat_inds_selected , lon_inds_selected, xmin, xres, ymax, yres, dimy, dimx):
        lats_per_index = np.arange(ymax, ymax + dimy * yres, yres)[lat_inds_selected]
        lons_per_index = np.arange(xmin, xmin + dimx * xres, xres)[lon_inds_selected]
        return lats_per_index, lons_per_index

    def transform(self, df, y=None):
        print("Filling presence and absence data.")
        data = gdal.Open(self.filename, gdal.GA_ReadOnly)
        geoinfo = data.GetGeoTransform()
        xmin, xres, ymax, yrot, xrot, yres = (
            geoinfo[0], geoinfo[1], geoinfo[3],
            geoinfo[4], geoinfo[2], geoinfo[-1]
        )
        array = data.ReadAsArray()
        print(f"Data for species {self.species} loaded from a file.")
        # array is too big; lets select every second point of it
        array = array[::2, ::2]
        yres*=2
        xres*=2

        all_presence_indices = np.argwhere(np.isin(array, self.presence_values))

        selected_presence_inds = all_presence_indices[
            np.random.choice(all_presence_indices.shape[0], self.num_to_select, replace=False)
        ]
        print("Absence inds selected...")
        lat_inds_selected = selected_presence_inds[:, 0]
        lon_inds_selected = selected_presence_inds[:, 1]
        dimx = array.shape[1]
        dimy = array.shape[0]
        lats_presence, lons_presence = (
            self.convert_inds_to_coords(lat_inds_selected , lon_inds_selected, xmin, xres, ymax, yres, dimy, dimx)
        )

        presence_df = pd.DataFrame(
            {"species": self.species, "latitude": lats_presence, "longitude": lons_presence, "absence": False}
        )
        del selected_presence_inds, all_presence_indices
        print("Presence data formed.")
        all_absence_indices = np.argwhere(~np.isin(array, self.presence_values) & (array != self.nan_value) & (array > 0))
        del array

        selected_absence_inds = all_absence_indices[
            np.random.choice(all_absence_indices.shape[0], self.num_to_select, replace=False)
        ]
        del all_absence_indices

        lat_inds_selected = selected_absence_inds[:, 0]
        lon_inds_selected = selected_absence_inds[:, 1]
        lats_absence, lons_absence = (
            self.convert_inds_to_coords(lat_inds_selected , lon_inds_selected, xmin, xres, ymax, yres, dimy, dimx)
        )

        absence_df = pd.DataFrame(
            {"species": self.species, "latitude": lats_absence, "longitude": lons_absence, "absence": True}
        )
        print("Absence data formed.")
        del selected_absence_inds
        df_ = df.copy()
        df_['absence'] = False
        df_ = pd.concat([df_, presence_df, absence_df], ignore_index=True)
        df_.drop_duplicates(["species", "latitude", "longitude", "absence"], ignore_index=True, inplace=True)
        print(f"Presence and absence data were merged. shape={df_.shape}")
        return df_


class PruneSuspiciousCoords(PreprocessingMixin):
    '''
    Removes all records having suspicious precision of positioning.
    If latitute and longitude of a record have the number
    of floating digits less than <digits> ,it is treated as suspicious.
    '''

    def __init__(self, digits=2):
        self.digits_ = digits

    def transform(self, df, y=None):
        val = 10 ** self.digits_
        lats_ind = np.equal(np.mod(df.latitude.values * val, 1), 0)
        lons_ind = np.equal(np.mod(df.longitude.values * val, 1), 0)
        return df.copy()[~(lats_ind + lons_ind)]


class ExpertFeatureSelector(PreprocessingMixin):
    '''
    Feature selection utility function.

    Select only those featrues that were selected by an expert.
    '''

    def __init__(self, variables=None):
        self.vartiables_ = variables

    def transform(self, df, y=None):
        if not self.variables_: return df

        filtered = []
        for name in self.variables_:
            if name in df:
                filtered.append(name)
        return df[filtered]


class CorrelationPruner(PreprocessingMixin):
    def __init__(self, threshold=0.9, variables=[]):
        self.threshold_ = threshold
        self.variables_ = variables

    def transform(self, df, y=None):
        corrmatrix = df.loc[:, self.variables_].corr()
        res_vars = []
        lost = set(self.variables_)
        for var in self.variables_:
            if var in lost:
                res_vars.append(var)
                lost = lost - set({var})
            for toremove in lost:
                if abs(corrmatrix.loc[var, toremove]) >= self.threshold_:
                    lost = lost - set({toremove})
        removed = list(set(self.variables_) - set(res_vars))
        return df.copy().drop(removed, axis=1)


class FillPseudoAbsenceData(PreprocessingMixin):
    def __init__(self, density=0.1, area=None):
        '''
        Fill data frame with pseudo-absence data
        '''
        self.density_ = float(density)
        self.area_ = area

    def update_df(self, df, ar, sp):
        size = abs((ar[0] - ar[-2]) * (ar[-1] - ar[1]))
        num = int(size * self.density_)
        lats = np.random.uniform(ar[0], ar[2], num)
        lons = np.random.uniform(ar[1], ar[-1], num)
        res = pd.concat([df, pd.DataFrame({'species': [sp] * len(lats),
                                            'latitude': lats,
                                            'longitude': lons,
                                            'absence': [True] * len(lats)})])
        return res

    def transform(self, df, y=None):
        res = df.copy()
        assert len(df.species.unique()) == 1, "DataFrame should contain only one species"
        sp = df.species.unique()[-1]
        if 'absence' not in res.columns:
            res['absence'] = False
        if self.area_ is None:
            if sp in absence_data:
                for ar in absence_data[sp]:
                    res = self.update_df(res, ar, sp)
            if 'all' in absence_data:
                for ar in absence_data['all']:
                    res = self.update_df(res, ar, sp)
        else:
            res = self.update_df(res, self.area_, sp)
        res.absence = res.absence.astype(np.bool)
        return res


class FillPseudoAbsenceByConditions(PreprocessingMixin):
    def __init__(self, similarity=1, density=0.1, species='', area=None):
        self.similarity_ = similarity
        self.density_ = float(density)
        self.species_ = species
        self.area_ = area

    def transform(self, df, y=None):
        sselect = SelectSpecies(self.species_)
        df_ = sselect.fit_transform(df)
        if self.area_ is not None:
            lat_min, lat_max = self.area_[0][0], self.area_[1][0]
            lon_min, lon_max = self.area_[0][1], self.area_[1][1]
        else:
            lat_min, lat_max =  min(df_.latitude), max(df_.latitude)
            lon_min, lon_max = min(df_.longitude), max(df_.longitude)
        size = (lat_max - lat_min) * (lon_max - lon_min)
        num = int(size * self.density_)
        lats_candidates = np.random.uniform(lat_min, lat_max, num)
        lons_candidates = np.random.uniform(lon_min, lon_max, num)
        variables = list(set(df_.columns.values) - set(['latitude','longitude',
                                                  'species', 'absence']))
        df_cand = pd.DataFrame({'latitude': lats_candidates, 'longitude': lons_candidates})
        filler = FillEnvironmentalData(variables=variables)
        data_candidates = filler.fit_transform(df_cand)
        data_presence = df_.loc[:, variables].values
        candidate_values = data_candidates.loc[:, variables].values
        candidate_values /= np.std(candidate_values, axis=0)
        data_presence /= np.std(data_presence, axis=0)
        res = cdist(candidate_values, data_presence)
        threshold = float(len(variables)) * self.similarity_
        inds = np.all(res > threshold, axis=1)
        data_candidates = data_candidates[inds]
        data_candidates['absence'] = True
        data_candidates['species'] = self.species_
        print("The number of ps-absence by cond:", np.sum(inds))
        res = pd.concat([df, data_candidates])
        res.absence = res.absence.astype(np.bool)
        return res.dropna().reset_index(drop=True)

class FillEnvironmentalData(PreprocessingMixin):

    def __init__(self, variables=None, postfix=''):
        self.variables_ = variables
        self.postfix_ = postfix

    def transform(self, df, y=None):
        df_ = df.copy()
        for var in self.variables_:
            values = get_predictor_data(tuple(df_['latitude'].values),
                                        tuple(df_['longitude'].values), var,
                                        postfix=self.postfix_)
            df_[var] = values
        return df_.dropna().reset_index(drop=True)

    def transform_nans(self, df, y=None):
        df_ = df.copy()
        for var in self.variables_:
            values = get_predictor_data(tuple(df_['latitude'].values),
                                        tuple(df_['longitude'].values), var,
                                        postfix=self.postfix_)
            df_[var] = values
        return df_


class SelectSpecies(PreprocessingMixin):

    def __init__(self, species):
        self.species_ = species

    def transform(self, df, y=None):
        df_ = df.copy()
        df_ = df_[df.species.str.contains(self.species_)]
        df_.species = self.species_
        print("Species selected...")
        return df_.reset_index(drop=True)


class SelectSpeciesList(PreprocessingMixin):
    def __init__(self, splist, overwrite=False):
        self.splist_ = splist
        self.overwrite = overwrite

    def transform(self, df, y=None):
        df_ = pd.DataFrame()
        for sp in self.splist_:
            intermediate = df[df.species.str.contains(sp)]
            if self.overwrite:
                intermediate.species = sp
            df_ = pd.concat([df_, intermediate])

        return df_.reset_index(drop=True)


class TreeFeatureImportance(PreprocessingMixin):

    def __init__(self, variables=[], iterations=10, nest=200, pabs_density=0.1):
        self.iterations_=iterations
        self.nest_ = nest
        self.variables_ = tuple(variables)
        self.pabs_density_ = pabs_density

    def transform(self, df, y=None):
        self.df_ = df.copy()
        variables = list(self.variables_)
        common = Pipeline([('fillabsence', FillPseudoAbsenceData(density=self.pabs_density_)),
                           ('fillenvironmental', FillEnvironmentalData(variables)),
                           ])
        forest = ExtraTreesClassifier(n_estimators=self.nest_,
                                      random_state=0)
        importances = []
        print("Performing iterations of the feature selection procedure.")
        for i in range(self.iterations_):
            print('Current iteration is ', i)
            new_df = common.fit_transform(self.df_)
            y = new_df['absence'].values
            X = new_df[variables].values
            forest.fit(X, y)
            importances.append(forest.feature_importances_)
        print("Iterations finished.")
        self.feature_importances_ = np.mean(np.array(importances), axis=0).tolist()
        return df


class RFECV_FeatureSelector(PreprocessingMixin):

    def __init__(self, clfs=[('MaxEnt', LogisticRegression()),
                             ], cv=StratifiedKFold(5),
                 score_func='accuracy'):
        self.clfs_ = clfs
        self.score_func = score_func
        self.cv_ = cv

    def transform(self, df, y=None):
        result = dict()
        for name, clf in zip(clfs):
            estimator = RFECV(clf, step=1, cv=self.cv_,
                              njobs=3)


def get_probabilities(LATS, LONS, clf, optimal_vars, postfix=''):
    LATS_GRID, LONS_GRID = np.meshgrid(LATS, LONS)
    fill_env_data = FillEnvironmentalData(optimal_vars, postfix)
    map_df = pd.DataFrame({'latitude': LATS_GRID.ravel(),
                           'longitude': LONS_GRID.ravel()
                           })
    filled_df = fill_env_data.transform_nans(map_df)
    XMAP = filled_df.loc[:, optimal_vars].values
    nan_mask = np.any(np.isnan(XMAP), axis=1)
    predictions = np.zeros((len(nan_mask), 2)) * np.nan
    predictions[~nan_mask, :] = clf.predict_proba(XMAP[~nan_mask, :])
    presence_proba_current = predictions[:, 1]
    # if not postfix:
    #     response['proba'].extend(predictions[~nan_mask, 1].tolist()[::4])
    #     for var in optimal_vars:
    #         response[var].extend(filled_df.loc[~nan_mask, var].values.tolist()[::4])
    return presence_proba_current.reshape(LATS_GRID.shape).T


def plot_map(lat_range, lon_range, resolution, clf, optimal_vars, train_df=None,
             name='', postfix=''):
    # response = collections.defaultdict(list)

    LONS = np.linspace(*lon_range, resolution)
    if resolution <= 5000:
        LATS = np.linspace(*lat_range, resolution)
        presence_proba_current = get_probabilities(LATS, LONS, clf,
                                                   optimal_vars, postfix)
    else:
        split_k = 1
        done = False
        for k in range(2, 100):
            for j in [0,1,-1,2,-2]:
                r = resolution + j
                if r % k == 0 and r / k * resolution <= 1.0e+6:
                    split_k = k
                    done = True
                    break
            if done: break

        LATS = np.split(np.linspace(*lat_range, resolution), split_k)
        result =[]

        for ind, lats in tqdm(enumerate(LATS)):
            _ = get_probabilities(lats, LONS, clf, optimal_vars, postfix)
            result.append(_)
        presence_proba_current = np.vstack(result)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ccmap = plt.get_cmap('CMRmap')
    ccmap.set_bad(color='#12680c')
    cf = ax.imshow(presence_proba_current,
                   cmap=ccmap, origin='lower',
                   extent=list(lon_range) + list(lat_range))
    fig.colorbar(cf, orientation='vertical', ticks=np.linspace(0,1,20))
    ax.set_title('%s:' % name)
    if train_df is not None:
        psedo_absence_lats = train_df[train_df.absence == True].latitude.values
        pseudo_absence_lons = train_df[train_df.absence == True].longitude.values
        presence_lats = train_df[train_df.absence == False].latitude.values
        presence_lons = train_df[train_df.absence == False].longitude.values
        #ax.plot(pseudo_absence_lons, psedo_absence_lats, 'rx', markersize=2)
        ax.plot(presence_lons, presence_lats, 'r.', markersize=2)

    #Save results to a tiff-file
    array_to_raster(presence_proba_current[::-1], lat_range,
                    lon_range, '%s_%s' % ('_'.join(name.split('_')[:-1]), name.split('_')[-1])  + '.tiff')


    # if not postfix:
    #     datname = '%s' % '_'.join(name.split('_')[:-1])  + '.dat'
    #     with open(datname, 'wb') as f:
    #         pickle.dump(response, f)
    return fig, ax
