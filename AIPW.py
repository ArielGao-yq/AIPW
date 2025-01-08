import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.special
from econml.metalearners import TLearner
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import shap
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm

class AIPW_learner:
    def __init__(self,X,T,Y,
                    learner="Tlearner",
                    confidence_intervel_level=0.95,
                    trim_level=0,
                    num_of_fold=5,
                    CI_method="CLT",
                    n_bootstrap=100):
        #args: learner: the base learner to use
        self.learner = learner
        self.confidence_intervel_level = confidence_intervel_level
        self.trim_level = trim_level
        self.num_of_fold = num_of_fold #cv fold
        self.CI_method =CI_method
        self.n_bootstrap = n_bootstrap #when n_bootstrap is too small, the CI may not contain the point estimate of ATE
        self.p_hat=None

    def single_fit(self,modelType):
        # Fit a single model based on the model type
        if modelType == "regression":
            model = LGBMRegressor()
        else:
            model = LGBMClassifier()

        return model  # Return the fitted model

    def cross_fit(self, X, T,Y, modelType="regression"):
        index_T = T.index[T == 1]
        index_C = T.index[T == 0]

        # Initialize Series to store predictions
        Y1_hat_series = pd.DataFrame(index=X.index, columns=range(self.num_of_fold))
        Y0_hat_series =  pd.DataFrame(index=X.index, columns=range(self.num_of_fold))

        if modelType == "regression":
            # Split for treatment and control groups
            X_T = X.loc[index_T]
            Y_T = Y.loc[index_T]
            X_C = X.loc[index_C]
            Y_C = Y.loc[index_C]

            kf_T = KFold(n_splits=self.num_of_fold, shuffle=True, random_state=42)
            kf_C = KFold(n_splits=self.num_of_fold, shuffle=True, random_state=42)

            shap_values_Y1 = np.zeros((X.shape[0], X.shape[1]))
            shap_values_Y0 = np.zeros((X.shape[0], X.shape[1]))


            # Use treatment group to fit Y1

            for fold, (train_index, test_index) in enumerate(kf_T.split(X_T)):
                X_train, X_test = X_T.iloc[train_index], X_T.iloc[test_index]
                Y_train, Y_test = Y_T.iloc[train_index], Y_T.iloc[test_index]

                model = self.single_fit(modelType)
                model.fit(X=X_train, y=Y_train)
                Y1_hat = model.predict(X)
                original_train_index = X_T.index[train_index]  # Get original indices for training data


                Y1_hat[original_train_index] = np.nan  # Set training predictions to NaN

                # Convert Y1_hat to pandas Series to ensure index alignment with X
                Y1_hat_series[fold] = pd.Series(Y1_hat, index=X.index)

                # SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                shap_values_Y1 += shap_values

            # Use control group to fit Y0
            for fold, (train_index, test_index) in enumerate(kf_C.split(X_C)):
                X_train, X_test = X_C.iloc[train_index], X_C.iloc[test_index]
                Y_train, Y_test = Y_C.iloc[train_index], Y_C.iloc[test_index]

                model = self.single_fit(modelType)
                model.fit(X=X_train, y=Y_train)
                Y0_hat = model.predict(X)
                original_train_index = X_C.index[train_index]
                Y0_hat[original_train_index] = np.nan  # Set training predictions to NaN

                # Convert Y0_hat to pandas Series to ensure index alignment with X
                Y0_hat_series[fold] = pd.Series(Y0_hat, index=X.index)

                # SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                shap_values_Y0 += shap_values

            # Average predictions
            cf_Y0_hat = np.nanmean(Y0_hat_series, axis=1)
            cf_Y1_hat = np.nanmean(Y1_hat_series, axis=1)

            # Combine SHAP values for regression
            # shap_values_Y0 = np.vstack(shap_values_Y0)
            # shap_values_Y1=np.vstack(shap_values_Y1)
            shap_values_Y0=shap_values_Y0/self.num_of_fold
            shap_values_Y1=shap_values_Y1/self.num_of_fold
            return cf_Y0_hat, cf_Y1_hat, shap_values_Y0, shap_values_Y1,Y0_hat_series,Y0_hat,train_index

        else:

            propensity_score = pd.Series(index=Y.index, dtype=float).fillna(np.nan)
            kf = KFold(n_splits=self.num_of_fold, shuffle=True, random_state=10)

            shap_values_classify =  np.zeros((X.shape[0], X.shape[1]))


            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
                T_train, T_test = T.iloc[train_index], T.iloc[test_index]

                model = self.single_fit(modelType)
                model.fit(X_train, T_train)

                propensity = model.predict_proba(X_test)[:, 1]  # Get probability of T=1
                propensity_score[test_index] = propensity

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                shap_values_classify+=shap_values[1]

            shap_values_classify=shap_values_classify/self.num_of_fold
            cf_propensity_score = propensity_score.groupby(propensity_score.index).mean()

            # Combine SHAP values for classification
            #shap_values_classify = np.vstack(shap_values_classify)


            return cf_propensity_score, shap_values_classify
    def propensity_score(self,X,T,Y):
        p_score,shap_values_classify=self.cross_fit(X,T,Y,modelType="classfication")
        AUC = roc_auc_score(T, p_score)
        return p_score, AUC, shap_values_classify


    def AIPW_individual_effect(self,X,T,Y):
        cf_Y0_hat,cf_Y1_hat, shap_values_Y0, shap_values_Y1,Y0_hat_series,Y0_hat,train_index= self.cross_fit(X,T,Y,modelType="regression")
        self.p_hat, self.AUC, shap_values_classify=self.propensity_score(X,T,Y)
        #estimat AIPW individual treatment effect for each individual
        est_individual_effect = cf_Y1_hat + T * (Y - cf_Y1_hat) / (self.p_hat) - (cf_Y0_hat + (1 - T) * (Y - cf_Y0_hat) / (1 - self.p_hat))
        return est_individual_effect, shap_values_Y0, shap_values_Y1,shap_values_classify,Y0_hat_series,Y0_hat,train_index,cf_Y0_hat

    def ate_fit(self,X,T,Y,index=None):
        if index is None:
            index=X.index
        individual_effect = self.AIPW_individual_effect(X, T, Y)[0].loc[index]
        if self.trim_level == 0:
            ate = individual_effect.mean()
        else:
            self.p_hat = self.p_hat
            trimmed_l = np.percentile(self.p_hat, self.trim_level * 100)
            trimmed_u = np.percentile(self.p_hat, (1 - self.trim_level) * 100)
            trim_index = self.p_hat[(self.p_hat > trimmed_l) & (self.p_hat < trimmed_u)].index
            index=index.intersection(trim_index)
            ate = self.individual_effect.iloc[index].mean()
        return ate

    def ate_bootstrap(self, X, T, Y,index=None):
        if index is None:
            index=X.index
        # construct confidence interval for ATE by bootstrapping
        bootstrap_estimates = np.zeros(self.n_bootstrap)

        # Perform bootstrap resampling
        n = len(Y)

        for i in range(self.n_bootstrap):
            # Resample indices with replacement
            indices = np.random.choice(range(n), size=n, replace=True)
            X_boot = X.iloc[indices].reset_index(drop=True)
            T_boot = T.iloc[indices].reset_index(drop=True)
            Y_boot = Y.iloc[indices].reset_index(drop=True)

            # Calculate the ATE for the bootstrapped sample
            bootstrap_estimates[i] = self.ate_fit(X_boot, T_boot, Y_boot,index)
        return bootstrap_estimates

    def fit(self,X,T,Y):
        self.individual_effect,self.shap_values_Y0, self.shap_values_Y1,self.shap_values_classify,self.Y0_hat_series,self.Y0_hat,self.train_index,self.cf_Y0_hat=self.AIPW_individual_effect(X,T,Y)

        if self.trim_level==0:
            ate=self.individual_effect.mean()
        else:
            self.p_hat=self.propensity_score(X,T,Y)[0]
            self.trimmed_l=np.percentile(self.p_hat,self.trim_level/2*100)
            self.trimmed_u=np.percentile(self.p_hat,(1-self.trim_level/2)*100)
            trim_index=self.p_hat[(self.p_hat>=self.trimmed_l) & (self.p_hat<=self.trimmed_u)].index
            ate=self.individual_effect.iloc[trim_index].mean()

        if self.CI_method == "bootstrap":
            self.bootstrap_estimates = self.ate_bootstrap(X, T, Y)
            lower_bound = np.percentile(self.bootstrap_estimates, (1 - self.confidence_intervel_level) / 2 * 100)
            upper_bound = np.percentile(self.bootstrap_estimates, (1 + self.confidence_intervel_level) / 2 * 100)
        else:
            if self.trim_level==0:
                t_value = scipy.stats.t.ppf(1 - (1 - self.confidence_intervel_level) / 2,
                                            df=len(self.individual_effect) - 1)
                lower_bound = ate - t_value * np.std(self.individual_effect) / np.sqrt(len(self.individual_effect))
                upper_bound = ate + t_value * np.std(self.individual_effect) / np.sqrt(len(self.individual_effect))
            else:
                t_value = scipy.stats.t.ppf(1 - (1 - self.confidence_intervel_level) / 2,
                                            df=len(self.individual_effect) - 1)
                lower_bound = ate - t_value * np.std(self.individual_effect.iloc[trim_index]) / np.sqrt(len(self.individual_effect.iloc[trim_index]))
                upper_bound = ate + t_value * np.std(self.individual_effect.iloc[trim_index]) / np.sqrt(len(self.individual_effect.iloc[trim_index]))

        self.ate=ate
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound

        return ate,[lower_bound,upper_bound]
    def propensity_report(self):
        if self.p_hat is None:
            raise ValueError("Please fit the model first")
        return self.AUC,[self.p_hat]

    def individual_effect(self):
        if self.individual_effect is None:
            raise ValueError("Please fit the model first")
        return self.individual_effect
    def overlap_plot(self,X,T,Y,filename='overlap_plot.png'):
        index_T = T.index[T == 1]
        index_C = T.index[T == 0]
        prop_C=self.p_hat.loc[index_C]
        prop_T=self.p_hat.loc[index_T]
        trimmed_l = np.percentile(self.p_hat, self.trim_level*100)
        trimmed_u = np.percentile(self.p_hat, (1 - self.trim_level)*100)
        plt.figure(figsize=(10, 6))
        plt.hist(prop_C, bins=100, alpha=0.6, label='Control Group', color='blue')
        plt.hist(prop_T, bins=100, alpha=0.6, label='Treatment Group', color='orange')
        plt.axvline(x=trimmed_l, color='black', linestyle='--',label="trim level=%f"%np.round(self.trim_level,2))
        plt.axvline(x=trimmed_u, color='black', linestyle='--')
        # Adding titles and labels
        plt.title('Histogram of Propensity Scores')
        plt.xlabel('Propensity Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()

        # Save the plot to a file
        plt.savefig(filename)
        plt.close()


    def shap_plot(self,X,T,Y):

        with PdfPages('shap_plots.pdf') as pdf:
            # First plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values_Y0,X,show=False)
            plt.title('SHAP Summary Plot E[Y|T=0,X]')
            pdf.savefig()  # Save current figure to the PDF
            plt.close()

            # Second plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values_Y1, X,show=False)
            plt.title('SHAP Summary Plot E[Y|T=1,X]')
            pdf.savefig()  # Save current figure to the PDF
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values_classify, X,show=False)
            plt.title('SHAP Summary Plot E[T=1|X]')
            pdf.savefig()  # Save current figure to the PDF
            plt.close()

    def cate(self,X,T,Y,index=None,trim_level=0,CI_method="CLT",confidence_intervel_level=0.95):

        cate=self.ate_fit(X,T,Y,index)
        if index is None:
            index=X.index
        if CI_method=="bootstrap":
            self.bootstrap_estimates_cate=self.ate_bootstrap(X,T,Y,index)
            lb=np.percentile(self.bootstrap_estimates_cate,(1-confidence_intervel_level)/2*100)
            ub=np.percentile(self.bootstrap_estimates_cate,(1+confidence_intervel_level)/2*100)
        else:
            if trim_level == 0:
                t_value = scipy.stats.t.ppf(1 - (1 - confidence_intervel_level) / 2, df=len(self.individual_effect.iloc[index]) - 1)
                lb = cate - t_value * np.std(self.individual_effect.iloc[index]) / np.sqrt(len(self.individual_effect.iloc[index]))
                ub = cate + t_value * np.std(self.individual_effect.iloc[index]) / np.sqrt(len(self.individual_effect.iloc[index]))
            else:
                trimmed_l = np.percentile(self.p_hat, self.trim_level / 2 * 100)
                trimmed_u = np.percentile(self.p_hat, (1 - self.trim_level / 2) * 100)
                trim_index = self.p_hat[(self.p_hat > trimmed_l) & (self.p_hat < trimmed_u)].index
                index=index.intersection(trim_index)
                t_value = scipy.stats.t.ppf(1 - (1 - confidence_intervel_level) / 2,
                                            df=len(self.individual_effect.iloc[index]) - 1)
                lb = cate - t_value * np.std(self.individual_effect.iloc[index]) / np.sqrt(
                    len(self.individual_effect.iloc[index]))
                ub = cate + t_value * np.std(self.individual_effect.iloc[index]) / np.sqrt(
                    len(self.individual_effect.iloc[index]))

        return cate,[lb,ub]

    def trim_summary(self,X,T,Y):
        if self.trim_level==0:
            return "No trimming"
        else:
            trimmed_size=np.sum(self.p_hat[(self.p_hat<self.trimmed_l) | (self.p_hat>self.trimmed_u)])
            trimmed_percent=trimmed_size/X.shape[0]
            return trimmed_percent,[self.trimmed_l,self.trimmed_u]

    def trim_shap(self,X,T):
        if self.trim_level==0:
            return "No trimming"
        else:
            trimmed_l = np.percentile(self.p_hat, self.trim_level * 100)
            trimmed_u = np.percentile(self.p_hat, (1 - self.trim_level) * 100)
            trim = np.where((self.p_hat < self.trimmed_l) | (self.p_hat > self.trimmed_u), 1, 0)

            # Ensure 'trim' is a pandas Series with the same index as X
            trim = pd.Series(trim, index=X.index)
            model = LGBMClassifier()
            X_T=X.assign(t=T)
            model.fit(X_T, trim)
            explainer = shap.TreeExplainer(model)
            shap_values_trim = explainer.shap_values(X_T)[1]

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_trim, X_T, show=False)
            plt.title('SHAP Summary Plot E[trimmed|X,T]')
            plt.savefig('shap_trim.png')  # Save current figure to the PDF
            plt.close()

    def balance_plt(self,X,T,Y):
        p_hat = np.array(self.p_hat)
        coef = [None] * len(X.columns)
        weights = np.asarray(np.where(T == 1, 1 / p_hat, 1 / (1 - p_hat))).T
        CI_l = [None] * len(X.columns)
        CI_u = [None] * len(X.columns)
        coef_nw= [None] * len(X.columns)
        CI_l_nw = [None] * len(X.columns)
        CI_u_nw = [None] * len(X.columns)

        T_array = np.asarray(T)

        for i, feature in enumerate(X.columns):
            sm.add_constant(T_array)
            wls_model_w = sm.WLS(X[feature], sm.add_constant(T_array), weights=weights)
            wls_result_w = wls_model_w.fit()


            wls_model_w_nw = sm.WLS(X[feature], sm.add_constant(T_array))
            wls_result_w_nw = wls_model_w_nw.fit()

            coef[i] = wls_result_w.params[1]
            CI_l[i] = wls_result_w.conf_int(alpha=0.05, cols=None)[0][1]
            CI_u[i] = wls_result_w.conf_int(alpha=0.05, cols=None)[1][1]

            coef_nw[i] = wls_result_w_nw.params[1]
            CI_l_nw[i] = wls_result_w_nw.conf_int(alpha=0.05, cols=None)[0][1]
            CI_u_nw[i] = wls_result_w_nw.conf_int(alpha=0.05, cols=None)[1][1]

        CI_l = np.array(CI_l).flatten()
        CI_u = np.array(CI_u).flatten()
        coef = np.array(coef).flatten()

        CI_l_nw = np.array(CI_l_nw).flatten()
        CI_u_nw = np.array(CI_u_nw).flatten()
        coef_nw = np.array(coef_nw).flatten()

        # Sort the coefficients and corresponding feature names
        sorted_indices = np.argsort(coef)  # Indices that would sort coef

        # Apply sorting to coef, X.columns, CI_l, and CI_u
        sorted_coef = coef[sorted_indices]
        sorted_X_columns = np.array(X.columns)[sorted_indices]  # Reorder feature names
        sorted_CI_l = CI_l[sorted_indices]
        sorted_CI_u = CI_u[sorted_indices]

        # Calculate the lower and upper error bars
        yerr_lower = sorted_coef - sorted_CI_l  # Negative error (coefficient - lower bound)
        yerr_upper = sorted_CI_u - sorted_coef  # Positive error (upper bound - coefficient)

        # Sort the coefficients and corresponding feature names
        sorted_indices_nw = np.argsort(coef_nw)  # Indices that would sort coef

        # Apply sorting to coef, X.columns, CI_l, and CI_u
        sorted_coef_nw = coef[sorted_indices_nw]
        sorted_X_columns_nw = np.array(X.columns)[sorted_indices_nw]  # Reorder feature names
        sorted_CI_l_nw = CI_l[sorted_indices_nw]
        sorted_CI_u_nw = CI_u[sorted_indices_nw]

        # Calculate the lower and upper error bars
        yerr_lower = sorted_coef - sorted_CI_l  # Negative error (coefficient - lower bound)
        yerr_upper = sorted_CI_u - sorted_coef  # Positive error (upper bound - coefficient)

        yerr_lower_nw = sorted_coef_nw - sorted_CI_l_nw  # Negative error (coefficient - lower bound)
        yerr_upper_nw = sorted_CI_u_nw - sorted_coef_nw  # Positive error (upper bound - coefficient)
        if self.trim_level!=0:
            trim_index=self.p_hat[(self.p_hat>=self.trimmed_l) & (self.p_hat<=self.trimmed_u)].index
            X_trim=X.loc[trim_index]
            p_hat_trim = np.array(self.p_hat.loc[trim_index])
            coef_trim = [None] * len(X_trim.columns)
            T_trim = np.asarray(T.loc[trim_index])
            weights_trim = np.asarray(np.where(T_trim == 1, 1 / p_hat_trim, 1 / (1 - p_hat_trim))).T
            CI_l_trim = [None] * len(X_trim.columns)
            CI_u_trim = [None] * len(X_trim.columns)

            for i, feature in enumerate(X.columns):
                sm.add_constant(T_trim)
                wls_model_w = sm.WLS(X_trim[feature], sm.add_constant(T_trim), weights=weights_trim)
                wls_result_w = wls_model_w.fit()
                coef_trim[i] = wls_result_w.params[1]
                CI_l_trim[i] = wls_result_w.conf_int(alpha=0.05, cols=None)[0][1]
                CI_u_trim[i] = wls_result_w.conf_int(alpha=0.05, cols=None)[1][1]

            CI_l_trim = np.array(CI_l_trim).flatten()
            CI_u_trim = np.array(CI_u_trim).flatten()
            coef_trim = np.array(coef_trim).flatten()

            # Sort the coefficients and corresponding feature names
            sorted_indices_trim = np.argsort(coef_trim)  # Indices that would sort coef

            # Apply sorting to coef, X.columns, CI_l, and CI_u
            sorted_coef_trim = coef_trim[sorted_indices_trim]
            sorted_X_columns_trim = np.array(X_trim.columns)[sorted_indices]  # Reorder feature names
            sorted_CI_l_trim= CI_l_trim[sorted_indices_trim]
            sorted_CI_u_trim = CI_u_trim[sorted_indices_trim]

            # Calculate the lower and upper error bars
            yerr_lower_trim = sorted_coef_trim - sorted_CI_l_trim  # Negative error (coefficient - lower bound)
            yerr_upper_trim = sorted_CI_u_trim - sorted_coef_trim # Positive error (upper bound - coefficient)

        plt.figure(figsize=(10, 6))

        # Plot the coefficients as dots with error bars

        plt.errorbar(sorted_X_columns, sorted_coef,
                     yerr=[yerr_lower, yerr_upper],  # yerr should be 2D: negative and positive errors
                     fmt='o', color='red', ecolor='red', capsize=5, label='trim level=0')
        # plt.errorbar(sorted_X_columns_nw, sorted_coef_nw,
        #              yerr=[yerr_lower_nw, yerr_upper_nw],  # yerr should be 2D: negative and positive errors
        #              fmt='o', color='red', ecolor='red', capsize=5, label='original')
        if self.trim_level != 0:
            plt.errorbar(sorted_X_columns_trim, sorted_coef_trim,
                         yerr=[yerr_lower_trim, yerr_upper_trim],  # yerr should be 2D: negative and positive errors
                         fmt='o', color='blue', ecolor='blue', capsize=5, label='trim level=%f'%np.round(self.trim_level,2))

        plt.title('Regress feature on treatment')
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        # Display the legend
        plt.legend()
        plt.xticks(rotation=45,fontsize=1)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to fit everything
        plt.savefig('balance_plot.png')
        plt.show()
        return weights,sorted_X_columns,sorted_coef,sorted_CI_l,sorted_CI_u
