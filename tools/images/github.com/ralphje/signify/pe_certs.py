#!/usr/bin/env python3
'''uses Signify to parse PE File certificates'''

import re
import hashlib
from signify import exceptions as sig_except
from signify.authenticode import AuthenticodeFile

def get_certs_from_file(filepath):
    '''returns string containing certificate dump of file at filepath'''
    with open(filepath, "rb") as f:
        # create SignedPEFile object and dump relevant info
        pefile = AuthenticodeFile.from_stream(f)
        results = dump_certs(pefile)
    return results
    
    
def dump_certs(pefile):
    '''returns dict of certificate info from given PE file'''
    results = {}
    signed_data_list = []
    try:
        # iterate over signed data in PE file
        for signed_data in pefile.signed_datas:
            data_dict = {}
            
            # get signer information
            signer_dict = {}
            signer_dict['program_name'] = str(signed_data.signer_info.program_name)
            signer_dict['program_url'] = str(signed_data.signer_info.more_info)
            signer_dict['issuer'] = get_short_name(signed_data.signer_info.issuer)
            signer_dict['serial'] = hex(signed_data.signer_info.serial_number)
            
            # populate data_dict
            data_dict['signer'] = signer_dict
            data_dict['certificates'] = [cert_dump(cert) for cert in signed_data.certificates]
            
            # if exists, get countersigner information
            if signed_data.signer_info.countersigner:
                countersigner = signed_data.signer_info.countersigner
                counter_dict = {}
                try:
                    counter_dict['issuer'] = get_short_name(countersigner.signer_info.issuer)
                    counter_dict['serial'] = hex(countersigner.signer_info.serial_number)
                except:
                    counter_dict['issuer'] = get_short_name(countersigner.issuer)
                    counter_dict['serial'] = hex(countersigner.serial_number)
                data_dict['countersigner'] = counter_dict
            
            signed_data_list.append(data_dict)
            
    except sig_except.SignedPEParseError as e:
        return {"error": f"{e}"}
    
    # verify PE file signatures by walking the cert chain
    try:
        pefile.verify()
        is_valid = True
    except (sig_except.VerificationError, sig_except.AuthenticodeVerificationError):
        is_valid = False
    
    # populate results dict
    results['signed_data'] = signed_data_list
    results['valid'] = is_valid
    
    return results
    
def cert_dump(cert):
    '''returns a dictionary of certificate info'''
    cert_dict = {}
    cert_dict['short_name'] = get_short_name(cert.subject.dn)
    cert_dict['issuer'] = get_short_name(cert.issuer)
    cert_dict['serial'] = hex(cert.serial_number)
    cert_dict['subject'] = cert.subject.dn
    cert_dict['not_before'] = cert.valid_from.isoformat()
    cert_dict['not_after'] = cert.valid_to.isoformat()
    cert_dict['md5'] = hashlib.md5(cert.to_der).hexdigest()
    cert_dict['sha1'] = hashlib.sha1(cert.to_der).hexdigest()
    return cert_dict
    
    
def get_short_name(attr):
    '''return CN field of given certificate attribute'''
    attr = str(attr)
    if 'CN=' in attr:
        regex_obj = re.search('CN=(.+?),', attr) 
        if regex_obj:
            short_name = regex_obj.group(1)
        else:
            short_name = re.search('CN=(.+)', attr).group(1)
    elif 'OU=' in attr:
        short_name = re.search('OU=(.+?),', attr).group(1)
    elif 'O=' in attr:
        short_name = re.search('O=(.+?),', attr).group(1)
    else:
        short_name = "Unknown Source"
    return short_name
