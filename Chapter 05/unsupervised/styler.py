import pandas as pd

def style_dataframe(df: pd.DataFrame):
    return df.style.set_table_styles([
        # Header styling
        {"selector": "thead th", "props": [
            ("background-color", "#f2f2f2"),  
            ("color", "black"),              
            ("font-weight", "bold"),         
            ("border", "1px solid #ddd"),    
            ("text-align", "center")        
        ]},
        # Body styling
        {"selector": "tbody td", "props": [
            ("background-color", "white"),   
            ("color", "black"),              
            ("border", "1px solid #ddd"),    
            ("text-align", "center")         
        ]}
    ]).set_properties(**{
        "border-collapse": "collapse",      
        "font-size": "12px",                
        "font-family": "Arial, sans-serif" 
    })