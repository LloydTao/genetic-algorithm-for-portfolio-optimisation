import quantstats as qs


def portfolio_sharpe(daily_prices, weights):
    """Calculate the Sharpe index of a given portfolio composition.

    Args:
        daily_prices (array): The historic price data of all assets.
        weights (list): The weighting of each asset within the portfolio.

    Returns:
        int: The Sharpe index of the portfolio.
    """
    # Build a data frame of historic daily returns
    daily_returns = daily_prices.diff().iloc[1:]

    # Weight and average the returns based on portfolio compostion
    weighted_returns = daily_returns.mul(weights)
    mean_returns = weighted_returns.mean(axis=1)

    # Calculate the Sharpe ratio of the weighted portfolio
    return qs.stats.sharpe(mean_returns)
