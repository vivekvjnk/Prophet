'''
JOERN based code analyzer tool for Prophet
=========================================
# What can JOERN do ? (JOERN is based on SCALA language)
    1. Given a code base, JOERN can find all the functions/methods 
        - Here cpg_m_name = cpg.method.name("<method_name>")
        - cpg_m_name.<options> 
        - Dump method : get all implementations (cpg_m_name.dump)
        - Method location : get file information in which method is defined (cpg_m_name.location.filename)
        - Find all local variables of function and make a list of them (cpg_m_name.local.<options>)
        - Find all returns of method
        - Find all the callers of the method (cpg_m_name.caller)
        - Get path of method call upto the top level: cpg_m_name.repeat(_.caller)(_.emit).name.l
        - We can also get all callees of the method

    2. Types 
        - cpg_t_name = 
    Doctor, Stock exchange(quant), RCS 
'''