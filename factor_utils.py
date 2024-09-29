## UTILS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from typing import Optional, List, Dict, Any, Tuple, Union

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc,
    plot_confusion_matrix, plot_roc_curve
)

def procent(p, quote = False):
        if quote: 
            return '(' + str(round(p*100,2)) + '%)'
        

def utils_cdf(a: np.ndarray) -> Dict[str, List[float]]:
    """
    Computes the cumulative distribution function (CDF) of a given array `a`.

    The CDF is calculated by sorting the input array, and for each unique value in the array, 
    the cumulative probability is updated. The function returns a dictionary with two keys:
    - 'x': The sorted values from the input array, repeated to form the step function.
    - 'cdf': The cumulative distribution function values corresponding to 'x'.

    Args:
    a (np.ndarray): Input array of numerical values.

    Returns:
    Dict[str, List[float]]: A dictionary containing:
        - 'x': A list of the sorted values, repeated to form a step-like graph for the CDF.
        - 'cdf': A list of the cumulative probabilities corresponding to 'x'.
    """
    sorted_a: np.ndarray = np.sort(a)
    x2: List[float] = []  # Holds the repeated sorted values for the CDF step function
    y2: List[float] = []  # Holds the cumulative probability values
    y: float = 0.0  # Initial cumulative probability

    for x in sorted_a:
        x2.extend([x, x])  # Repeat each sorted value to create steps in the CDF
        y2.append(y)       # Append the current cumulative probability
        y += 1.0 / len(a)   # Increment the cumulative probability
        y2.append(y)       # Append the updated cumulative probability

    return {'x': x2, 'cdf': y2}


def get_alpha_mu(data: any, x1: float, x2: float, p1_logistic: float = 0.05, p2_logistic: float = 0.95) -> Optional[Dict[str, float]]:
    """
    Computes the parameters `alpha` and `mu` for a logistic function based on two given points 
    and their corresponding logistic probabilities.

    Args:
    data (any): An additional input, not used in the computation, for potential extensions.
    x1 (float): The first point along the x-axis.
    x2 (float): The second point along the x-axis (must be greater than `x1`).
    p1_logistic (float, optional): The logistic probability at `x1`. Must be in the range (0, 1). Default is 0.05.
    p2_logistic (float, optional): The logistic probability at `x2`. Must be in the range (0, 1). Default is 0.95.

    Returns:
    Optional[Dict[str, float]]: A dictionary containing:
        - 'alpha': The slope of the logistic function.
        - 'mu': The intercept of the logistic function.
    Returns `None` if input validation fails.
    
    Example usage:
        result = get_alpha_mu(data, 1.0, 2.0)
        if result:
            alpha = result['alpha']
            mu = result['mu']
    """

    # Validate p1_logistic
    if 0 < p1_logistic < 1:
        l1 = math.log(1 / p1_logistic - 1)
    else:
        print("[error] p1_logistic must be in (0, 1)")
        return None

    # Validate p2_logistic
    if 0 < p2_logistic < 1:
        l2 = math.log(1 / p2_logistic - 1)
    else:
        print("[error] p2_logistic must be in (0, 1)")
        return None

    # Ensure x2 is greater than x1
    if x2 > x1:
        alpha = (l1 - l2) / (x2 - x1)
        mu = (x1 * l2 - x2 * l1) / (l2 - l1)
        return {'alpha': alpha, 'mu': mu}
    else:
        print("[error] x2 must be greater than x1")
        return None

def logistic_transformation(
    factor_values: pd.Series, 
    x1: float, 
    x2: float, 
    p1_logistic: float = 0.05, 
    p2_logistic: float = 0.95
) -> Optional[Dict[str, Any]]:
    """
    Applies a logistic transformation to a set of factor values based on logistic function parameters.

    Args:
    factor_values (pd.Series): A pandas Series containing the factor values to be transformed.
    x1 (float): The first point along the x-axis used to fit the logistic function.
    x2 (float): The second point along the x-axis used to fit the logistic function (must be greater than `x1`).
    p1_logistic (float, optional): The logistic probability at `x1`. Must be in the range (0, 1). Default is 0.05.
    p2_logistic (float, optional): The logistic probability at `x2`. Must be in the range (0, 1). Default is 0.95.

    Returns:
    Optional[Dict[str, Any]]: A dictionary containing:
        - 'values': The transformed logistic factor values.
        - 'quant1': The original x1 value.
        - 'quant2': The original x2 value.
        - 'Middle_Point': The middle point of the logistic function.
        - 'Slope': The slope of the logistic function.
    Returns `None` if the logistic parameters cannot be calculated.
    
    Example usage:
        result = logistic_transformation(factor_values, 1.0, 2.0)
        if result:
            transformed_values = result['values']
    """

    # Convert factor values to numeric in case they aren't
    factor_values_temp: pd.Series = pd.to_numeric(factor_values)

    # Get logistic parameters (alpha and mu)
    alpha_mu: Optional[Dict[str, float]] = get_alpha_mu(factor_values, x1, x2, p1_logistic, p2_logistic)
    
    if alpha_mu is not None:
        # Extract alpha (Slope) and mu (Middle_Point)
        Slope: float = alpha_mu['alpha']
        Middle_Point: float = alpha_mu['mu']

        # Apply the logistic transformation to the factor values
        logistic_factor_value: pd.Series = 1 / (1 + np.exp(Slope * (Middle_Point - factor_values_temp)))

        # Prepare the result dictionary
        rez: Dict[str, Any] = {
            'values': logistic_factor_value,
            'quant1': x1,
            'quant2': x2,
            'Middle_Point': Middle_Point,
            'Slope': Slope
        }

        return rez
    
    # If alpha_mu is None, return None indicating an error in calculating logistic parameters
    return None

def isNaN(num):  
   return num != num
 
def calculate_fpr_tpr_gini_no_lib(y_true, y_scores):
    """
    Calculate Gini index, FPR (False Positive Rate), and TPR (True Positive Rate) with linear complexity
    based on correctly ordered pairs.
    
    :param y_true: List of true class labels (0 or 1).
    :param y_scores: List of predicted probabilities for the positive class.
    :return: Gini index, lists of FPR and TPR.
    """
    
    # Combine true labels and predicted scores, then sort by predicted scores in descending order
    combined = list(zip(y_true, y_scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    
    # Count the total number of positive and negative classes
    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives
    
    # Initialize variables for cumulative sums and correct pair counting
    cumulative_positives = 0
    cumulative_negatives = 0
    correct_pairs = 0
    
    # Initialize lists to store FPR and TPR values
    fpr_list = []
    tpr_list = []
    
    # Loop through sorted data
    for label, score in combined:
        if label == 1:
            cumulative_positives += 1
        else:
            cumulative_negatives += 1
            # For each negative example, add correct pairs as the number of positives seen before it
            correct_pairs += cumulative_positives
        
        # Calculate current TPR and FPR
        tpr = cumulative_positives / total_positives if total_positives != 0 else 0
        fpr = cumulative_negatives / total_negatives if total_negatives != 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Total number of pairs (P * N)
    total_pairs = total_positives * total_negatives
    
    # Calculate Gini index
    gini = (2 * correct_pairs) / total_pairs - 1 if total_pairs != 0 else 0
         
    return fpr_list, tpr_list, gini

def plot_factor(
    data2: pd.DataFrame, 
    factor_name: str = 'factorN', 
    value_col: str = 'FVALUE', 
    numeric: bool = True, 
    target_col: str = 'Default60', 
    nbins: int = 100, 
    p_low: float = 0.05, 
    p_high: float = 0.95, 
    left_cut: Optional[float] = None, 
    right_cut: Optional[float] = None, 
    show_cut_graph: bool = False
) -> None:
    """
    Plots several graphs to visualize the distribution and the performance of a factor in binary classification.

    The function plots a histogram of the factor values, a cumulative distribution function (CDF), a receiver 
    operating characteristic (ROC) curve, and compares the distributions for defaulted and non-defaulted cases.

    Args:
    data2 (pd.DataFrame): The dataset containing factor values and target labels.
    factor_name (str, optional): The name of the factor to be displayed in the plot title. Default is 'factorN'.
    value_col (str, optional): The column name for the factor values. Default is 'FVALUE'.
    numeric (bool, optional): Whether the factor values are numeric and should be transformed using logistic regression. Default is True.
    target_col (str, optional): The column name for the binary target variable (1/0). Default is 'Default60'.
    nbins (int, optional): Number of bins for the histograms. Default is 100.
    p_low (float, optional): The lower quantile for clipping the factor values. Default is 0.05.
    p_high (float, optional): The upper quantile for clipping the factor values. Default is 0.95.
    left_cut (Optional[float], optional): Value to cut the factor values from the left. Default is None.
    right_cut (Optional[float], optional): Value to cut the factor values from the right. Default is None.
    show_cut_graph (bool, optional): Whether to limit the graphs to the p_low-p_high quantile area. Default is False.

    Returns:
    None: The function plots the graphs and does not return anything.
    """ 
 
    def procent(p: float, quote: bool = False) -> str:
        return f"({round(p * 100, 2)}%)" if quote else str(round(p * 100, 2))   

    x = data2[value_col].copy()
    y = data2[target_col].copy()
    x_1 = x.loc[y == 1] #data_temp.loc[data_temp[target_col] == 1, value_col]
    x_0 = x.loc[y == 0] #data_temp.loc[data_temp[target_col] == 0, value_col]

    # Apply left and right cuts to the factor values
    if left_cut is not None:
        x.loc[x < left_cut] = left_cut
    if right_cut is not None:
        x.loc[x > right_cut] = right_cut  #data_temp.loc[data_temp[value_col] > right_cut, value_col] = right_cut

#stats
    print("Statistics->")
    nan_count = x.isna().sum()
    print(" nan values->" + str(nan_count)+ "  " +procent(nan_count / len(x), quote = True))
    inf_count = x.isin([np.inf, -np.inf]).sum().sum()
    print(" inf values->" + str(inf_count)+ "  " +procent(inf_count / len(x), quote = True))

#modifing
    if nan_count > 0:
        print(" na values deleting..")
        not_na_indexes = x.index.notna()  #notnull()
        x = x[not_na_indexes]
        y = y[not_na_indexes]
    if inf_count > 0:
        print(" inf values deleting..")        
        not_inf_indexes = x.np.isfinite(x).all(1)  #notnull()
        x = x[not_inf_indexes]
        y = y[not_inf_indexes]

    x_show = x
    x_1_show = x_1
    x_0_show = x_0

    min_x = min(x)
    max_x = max(x)
    print(f"min = {min_x}")
    print(f"max = {max_x}")

    low_quant = x.quantile(p_low)
    high_quant = x.quantile(p_high)

    if max_x == float('inf'):
        x.clip(upper=high_quant, inplace=True)
        print('maxFactorvalue-inf_clipped with high_quant')

    min_x = min(x)
    max_x = max(x)

    if show_cut_graph:
        x_show.clip(upper=high_quant, inplace=True)  
        x_show.clip(lower=low_quant, inplace=True)  
        x_0_show.clip(upper=high_quant, inplace=True)  
        x_0_show.clip(lower=low_quant, inplace=True) 
        x_1_show.clip(upper=high_quant, inplace=True)  
        x_1_show.clip(lower=low_quant, inplace=True) 

    min_x_show = min(x_show)
    max_x_show = max(x_show)
    print("after cutting: min ="+ str(min_x_show))
    print("after cutting: max ="+ str(max_x_show))

    print('Start plotting...')

    # Prepare figure and axes
    fig = plt.figure(figsize=(8, 12))
    
    # Graph 1: Histogram of factor values and defaults
    ax_0_0 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=1)
    bins = np.linspace(min_x_show, max_x_show, nbins + 1)
    ax_0_0.set_title(f"{factor_name}\n q_low={round(low_quant, 2)} {procent(p_low, True)} \n q_high={round(high_quant, 2)} {procent(p_high, True)}")
    ns1, _, _ = ax_0_0.hist(x_show, bins, alpha=0.5, label='all', rwidth=0.7)
    ns2, _, _ = ax_0_0.hist(x_1_show, bins, alpha=0.5, label='defaults', color='tab:red', rwidth=0.7)
    ax_0_0.legend(loc='upper right')
    ax_0_0.axvline(x=low_quant, color='red')
    ax_0_0.axvline(x=high_quant, color='red')

    # Add line for default rate
    z = ns2 / (ns1 + 0.01)
    mids = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]
    ax2 = ax_0_0.twinx()
    ax2.plot(mids, z, linewidth=1.5, label='Default Rate', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add logistic transformation line if the factor is numeric
    if numeric:
        logistic_factor = logistic_transformation(bins, low_quant, high_quant)['values']
        ax0_ = ax_0_0.twinx()
        ax0_.plot(bins, logistic_factor, color='blue', lw=2)
        ax0_.set_yticks([])

    # Graph 2: Histogram with density and cumulative distribution comparison
    ax_1_0 = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=1)
    bins6 = np.linspace(min_x_show, max_x_show, nbins + 1)
    ns3, _, _ = ax_1_0.hist(x_show, bins6, alpha=0.5, label='all', rwidth=1, density=True)
    ns4, _, _ = ax_1_0.hist(x_1_show, bins6, alpha=0.5, label='defaults', color='tab:red', rwidth=1, density=True)
    ax_1_0.legend(loc='upper right')
    ax_1_0.axvline(x=low_quant, color='red')
    ax_1_0.axvline(x=high_quant, color='red')

    # Cumulative distribution comparison
    scores_defaulted = x_1_show.copy().sort_values()
    scores_non_defaulted = x_0_show.copy().sort_values()
    scores_defaulted_cdf = utils_cdf(scores_defaulted)
    scores_non_defaulted_cdf = utils_cdf(scores_non_defaulted)
    ax_1_0.plot(scores_defaulted_cdf['x'], scores_defaulted_cdf['cdf'], linewidth=1.5, color='tab:red')
    ax_1_0.plot(scores_non_defaulted_cdf['x'], scores_non_defaulted_cdf['cdf'], linewidth=1.5, color='tab:blue')

    # Graph 3: Cumulative distribution function (CDF)
    ax_2_0 = plt.subplot2grid((3, 2), (2, 0), colspan=1, rowspan=1)
    x_sorted_show = x_show.sort_values()
    x_cdf_temp = list(range(1, len(x_sorted_show) + 1))
    x_cdf = [curr / x_cdf_temp[-1] for curr in x_cdf_temp]
    #ax3.set_xlim(min_x, max_x)
    ax_2_0.set_xbound(min_x_show, max_x_show)
    ax_2_0.plot(x_sorted_show, x_cdf, color='blue', lw=2, drawstyle='steps')
    ax_2_0.axvline(x=low_quant, color='red')
    ax_2_0.axvline(x=high_quant, color='red')
    ax_2_0.set_title(f'min={round(min_x_show, 2)} max={round(max_x_show, 2)}', fontsize=8)

    # Graph 4: ROC curve
    try:
        ax_2_1 = plt.subplot2grid((3, 2), (2, 1), colspan=1, rowspan=1)    
        print(" min x for ROC " + str(min(x))+ " max x for ROC " + str(max(x)))   
        print(" min y for ROC " + str(min(y))+ " max x for ROC " + str(max(y)))   
        print(" nan x for ROC " + str(x.isin([np.inf, -np.inf]).sum().sum())+ " nan x for ROC " + str(y.isin([np.inf, -np.inf]).sum().sum()))   
        #print(" nan y for ROC " + str(min(y))+ " nan x for ROC " + str(max(y)))  
        #fpr, tpr, _ = roc_curve(y, x)
        #gini = 2 * roc_auc_score(y, x) - 1
        fpr, tpr, gini = calculate_fpr_tpr_gini_no_lib(y, x)
        ax_2_1.set_xlabel('False Positive Rate')
        ax_2_1.set_ylabel('True Positive Rate')
        ax_2_1.set_title(f'ROC   gini={round(gini * 100, 2)}%', fontsize=8)
        ax_2_1.plot(fpr, tpr, color='darkorange', lw=2)
        ax_2_1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-')
    except:
        print("")
    plt.show()
    #return {'info':{'factor_name':factor_name, 'gini':gini, 'min_x':min_x, 'max_x':max_x,'errors': errorslnfo} }

def calculate_woe_iv(
    dataset: pd.DataFrame, 
    feature: str, 
    target: str
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a given feature in the dataset.

    The function generates a summary table that shows WoE, IV, and other key metrics for each unique value of the feature.
    It also calculates the probability of default (PD) for each value and provides cumulative distributions.

    Args:
    dataset (pd.DataFrame): The dataset containing the feature and target columns.
    feature (str): The column name of the feature for which WoE and IV are to be calculated.
    target (str): The column name of the binary target variable (1/0).

    Returns:
    Tuple[pd.DataFrame, float]: 
        - A DataFrame containing the WoE, IV, PD, and other metrics for each unique value of the feature.
        - The total Information Value (IV) for the feature.
    """

    # List to store the values for each unique feature value
    lst = []
    
    # Iterate over each unique value in the feature
    for val in dataset[feature].unique():
        all_count = dataset[dataset[feature] == val].count()[feature]
        good_count = dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature]
        bad_count = dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        
        # Append the results to the list
        lst.append({
            'Value': val,
            'All': all_count,
            'Good': good_count,
            'Bad': bad_count
        })
    
    # Create a DataFrame from the list
    dset = pd.DataFrame(lst)
    
    # Calculate distributions of Good and Bad for WoE and IV
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    
    # Calculate Weight of Evidence (WoE)
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    
    # Replace infinity values (for when Distr_Bad or Distr_Good is zero)
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    
    # Calculate Information Value (IV)
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    
    # Calculate the probability of default (PD)
    total_good_bad = dset['Good'].sum() + dset['Bad'].sum()
    dset['PD'] = dset['Bad'] / total_good_bad
    
    # Calculate total IV for the feature
    iv = dset['IV'].sum()
    
    # Sort values by the total count (All) in descending order
    dset.sort_values(by='All', ascending=False, inplace=True)
    
    # Calculate percentage and cumulative percentage for the total count (All)
    dset['All_proc'] = dset['All'] / dset['All'].sum()
    dset['All_cumm'] = dset['All'].cumsum()
    dset['All_cumm_proc'] = dset['All'].cumsum() / dset['All'].sum()
    
    return dset, iv

def plot_iv(df: pd.DataFrame, iv: float, factor_name: str) -> None:
    """
    Plots the Information Value (IV) for a given feature along with various graphs related to the feature's performance.

    Args:
    df (pd.DataFrame): DataFrame containing the WoE and IV information for the feature.
    iv (float): The calculated Information Value for the feature.
    factor_name (str): The name of the factor/feature being analyzed.

    Returns:
    None: The function plots the graphs and does not return any values.
    """
    fig = plt.figure(figsize=(8, 12))
    
    # First Graph: Distribution of all and default values with PD
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=1)
    x = df['All']
    y = df['Bad']
    vals = df['Value']

    max_len_string_to_show = 10

    def cut_string(x, max_len_string_to_show):
        if isinstance(x, str):
            rez = str(x[0:max_len_string_to_show])+ str("*" if len(x) > max_len_string_to_show else "" )
        else:
            rez = x
        return rez
    
    vals_to_show = [cut_string(x, max_len_string_to_show) for x in vals] 

    ax1.set_title(f'{factor_name}\n iv={round(iv * 100, 2)}%\n')
    x_range = range(len(vals))
    ax1.bar(x_range, x, width=0.5, label='all', tick_label=vals_to_show)
    ax1.bar(x_range, y, width=0.5, label='defaults', color='tab:red', tick_label=vals_to_show)
    ax1.legend(labels=['all', 'defaults'])
    
    # Plot PD on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x_range, df['PD'], color='red', lw=2)
    ax2.set_ylabel('PD')

    # Second Graph: WoE values with distribution bars
    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=1)
    ax3.bar(x_range, x, width=0.5, label='all', tick_label=vals_to_show)
    ax3.bar(x_range, y, width=0.5, label='defaults', color='tab:red', tick_label=vals_to_show)
    ax3.legend(labels=['all', 'defaults'])

    # Plot WoE on the secondary y-axis
    ax4 = ax3.twinx()
    ax4.plot(x_range, df['WoE'], color='red', lw=2)
    ax4.set_ylabel('WoE')

    # Third Graph: IV values with distribution bars
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2, rowspan=1)
    ax5.bar(x_range, x, width=0.5, label='all', tick_label=vals_to_show)
    ax5.bar(x_range, y, width=0.5, label='defaults', color='tab:red', tick_label=vals_to_show)
    ax5.legend(labels=['all', 'defaults'])

    # Plot IV on the secondary y-axis
    ax6 = ax5.twinx()
    ax6.plot(x_range, df['IV'], color='red', lw=2)
    ax6.set_ylabel('IV')

    plt.tight_layout()
    plt.show()

def categorical_plot(data: pd.DataFrame, name: str, target_feature: str) -> pd.DataFrame:
    """
    Plots a categorical variable against a target feature, showing the mean and cumulative count.

    Args:
    data (pd.DataFrame): The dataset containing the feature and target variable.
    name (str): The name of the categorical feature to be analyzed.
    target_feature (str): The target feature (binary classification variable).

    Returns:
    pd.DataFrame: A DataFrame summarizing the target feature's mean and cumulative count by the categorical feature.
    """
    df1 = data.groupby(by=name)[target_feature].describe().reset_index()
    df1.sort_values('mean', ascending=False, inplace=True)
    df1 = df1.reset_index(drop=True)
    
    # Calculate cumulative count
    df1['count_cumul'] = df1['count'].cumsum()
    target_mean = data[target_feature].mean()

    # Plot mean and cumulative count
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title(name)
    df1.plot(y=["mean"], figsize=(10, 5), ax=ax, grid=True)
    plt.axhline(y=target_mean, color='r', linestyle='-')  # Target mean as a reference line
    df1.plot(y='count_cumul', figsize=(10, 5), ax=ax, grid=True, secondary_y=True)
    
    plt.tight_layout()
    return df1

def print_score(true: pd.Series, pred: pd.Series, train: bool = True) -> None:
    """
    Prints the classification metrics including accuracy, classification report, and confusion matrix for the model's predictions.

    Args:
    true (pd.Series): The true labels.
    pred (pd.Series): The predicted labels by the model.
    train (bool): Flag to indicate whether the results are from the training set (True) or the test set (False).

    Returns:
    None: This function prints the evaluation metrics.
    """
    clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
    result_type = "Train" if train else "Test"
    
    print(f"{result_type} Result:\n{'='*48}")
    print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
    print(f" F1 score: {f1_score(true, pred) * 100:.2f}%")
    _, _, gini = calculate_fpr_tpr_gini_no_lib(true, pred)
    print(f" Gini: {gini * 100:.2f}%\n")

def evaluate_nn(true: Union[pd.Series, list], pred: Union[pd.Series, list], train: bool = True) -> None:
    """
    Evaluates a neural network model by printing classification metrics, including accuracy, classification report, and confusion matrix.

    Args:
    true (Union[pd.Series, list]): The true labels.
    pred (Union[pd.Series, list]): The predicted labels by the model.
    train (bool): Flag to indicate whether the results are from the training set (True) or the test set (False).

    Returns:
    None: This function prints the evaluation metrics.
    """
    clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
    result_type = "Train" if train else "Test"
    
    print(f"{result_type} Result:\n{'='*48}")
    print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
import matplotlib.pyplot as plt
from keras.callbacks import History

def plot_learning_evolution(r: History) -> None:
    """
    Plots the learning evolution for a training model, showing the loss and AUC score during training and validation phases.

    Args:
    r (History): The History object returned from the model fitting process (e.g., from Keras).

    Returns:
    None: This function creates plots showing the evolution of the model's loss and AUC score during training and validation.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot Loss evolution
    plt.subplot(2, 2, 1)
    plt.plot(r.history['loss'], label='Loss')
    plt.plot(r.history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot AUC evolution
    plt.subplot(2, 2, 2)
    plt.plot(r.history['AUC'], label='AUC')
    plt.plot(r.history['val_AUC'], label='Validation AUC')
    plt.title('AUC Score Evolution During Training')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

def make_optimal_cutoff_get_f1(y_real, predicted_probabilities, optimal_cutoff=None, verbose=False, text=""):
    """
    Finds the optimal cutoff for classifying probabilities and calculates the F1 score for predictions.
    
    Parameters:
    y_real (array-like): The actual labels (ground truth).
    predicted_probabilities (array-like): Predicted probabilities for the positive class (class 1).
    optimal_cutoff (float, optional): The cutoff threshold for classification. If None, the function calculates it 
                                      based on the actual default rate.
    verbose (bool, optional): If True, prints detailed information such as the cutoff, accuracy, and F1 score.
    text (str, optional): Additional text to print at the beginning of the function.

    Returns:
    tuple: A tuple containing:
        - f1 (float): The F1 score based on the predictions.
        - optimal_cutoff (float): The cutoff threshold used for classification.
        - y_pred (array-like): The predicted labels (0 or 1) based on the cutoff.
    """
    # Print the provided text if any (e.g., for logging or tracking purposes)
    print(text)

    # Step 1: Calculate the actual default rate (mean of the true labels in y_real)
    target_mean = np.mean(y_real)

    # Step 2: Find the optimal cutoff where the predicted default rate matches the actual default rate
    # Sort predicted probabilities in ascending order
    sorted_probs = np.sort(predicted_probabilities)

    # If no cutoff is provided, calculate it based on the actual default rate
    if not optimal_cutoff:
        # The optimal cutoff corresponds to the value at the index where the proportion of predictions
        # above this value matches the actual default rate
        optimal_cutoff = sorted_probs[int((1 - target_mean) * len(sorted_probs))]

    # If verbose, print the determined optimal cutoff
    if verbose:
        print(f"Optimal cutoff: {optimal_cutoff}")

    # Step 3: Apply the cutoff to classify probabilities
    # If a predicted probability is greater than or equal to the cutoff, classify as 1 (default), otherwise 0
    y_pred = (predicted_probabilities >= optimal_cutoff).astype(int)

    # Step 4: Evaluate performance

    # Accuracy (for reference, not returned by the function)
    accuracy = accuracy_score(y_real, y_pred)
    if verbose:
        print(f"Accuracy: {accuracy}")

    # Calculate the F1 score (harmonic mean of precision and recall) for the predictions
    f1 = f1_score(y_real, y_pred)
    if verbose:
        print(f"F1 Score: {f1}")

    # Return the F1 score, the optimal cutoff used, and the predicted labels
    return f1, optimal_cutoff, y_pred