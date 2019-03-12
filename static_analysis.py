import tensorflow as tf
import glob
from google.protobuf import text_format


def name_to_type_mapping(graph_def):
    mapping = {}
    for node in graph_def.node:
        mapping[node.name] = node.op
    return mapping

def get_ngVariables(graph_def):
    ngVar_name = []
    for node in graph_def.node:
        if node.op == "NGraphVariable":
            ngVar_name.append(node.name)
    return ngVar_name

def get_ngVariables_outputNode(graph_def, ngVar_name):
    all_nodes_has_ngVar_inputs = []
    for node in graph_def.node:
        for input_node in node.input:
            if input_node in ngVar_name:
                all_nodes_has_ngVar_inputs.append(node.op)
    return all_nodes_has_ngVar_inputs

# Extra copy added
def get_ngVariables_to_assign(graph_def, ngVar_name):
    ngVar_to_assign = 0
    for node in graph_def.node:
        if node.op == "Assign":
            for input_node in node.input:
                if input_node in ngVar_name:
                    ngVar_to_assign = ngVar_to_assign + 1
    return ngVar_to_assign

def get_ngVariables_to_apply(graph_def, ngVar_name):
    ngVar_to_apply = 0
    for node in graph_def.node:
        if node.op.startswith('Apply'):
            for input_node in node.input:
                if input_node in ngVar_name:
                    ngVar_to_apply = ngVar_to_apply + 1
    return ngVar_to_apply

def get_ngVariables_to_encap_saving(graph_def, ngVar_name):
    ngVar_to_encapsulate_saving = 0
    controlled_input = 0
    encap_input_not_supported = []
    encap_input_potential_support = []
    for node in graph_def.node:
        if node.op == "NGraphEncapsulate":
            for input_node in node.input:
                if input_node in ngVar_name:
                    ngVar_to_encapsulate_saving = ngVar_to_encapsulate_saving + 1
                elif input_node.startswith('^'):
                    controlled_input = controlled_input + 1
                elif input_node.startswith('ngraph'):
                    encap_input_potential_support.append((input_node, node.name))
                else:
                    encap_input_not_supported.append(input_node)

    return (ngVar_to_encapsulate_saving, controlled_input, encap_input_potential_support, encap_input_not_supported)



total_save_now = 0
total_potential_save = 0
total_extra_added = 0
# Find all the encapsulated_files
dump_graphs_folder = "./"
all_encapsulated_dump = glob.glob(dump_graphs_folder + "encapsulated_*.pbtxt")

for encapsulated_dump in all_encapsulated_dump:
    with open(encapsulated_dump) as f:
        txt = f.read()

    graph_def = text_format.Parse(txt, tf.GraphDef())

    name_to_op_mapping = name_to_type_mapping(graph_def)
    all_ngVars = get_ngVariables(graph_def)  # List of NGraphVariable names
    all_nodes_has_ngVar_inputs = get_ngVariables_outputNode(graph_def, all_ngVars)  # Set of nodes that has NGVariable input
    (ngVar_to_encapsulate_saving, controlled_input, encap_input_potential_support, encap_input_not_supported) = get_ngVariables_to_encap_saving(graph_def, all_ngVars)
    total_save_now = total_save_now + ngVar_to_encapsulate_saving
    print("Encapsulated dump %s -------------" %encapsulated_dump)
    print("Nodes has NGraphVariable as input ", set(all_nodes_has_ngVar_inputs))
    print("Number of NGVar -> Encap saving in one iteration ", ngVar_to_encapsulate_saving)
    print("Number of Controlled_inputs -> Encap ", controlled_input)
    print("Encap input potential support in one iteration ", encap_input_potential_support)
    total_potential_save = total_potential_save + len(encap_input_potential_support)
    encap_input_not_supported_op_type = []
    for node in encap_input_not_supported:
        encap_input_not_supported_op_type.append(name_to_op_mapping[node])
    print("Encap input not supported reason ", set(encap_input_not_supported_op_type))
    print("ngVar -> Apply copies that could save %d" %get_ngVariables_to_apply(graph_def, all_ngVars))
    total_potential_save = total_potential_save + get_ngVariables_to_apply(graph_def, all_ngVars)
    print("Extra copies added due to NGraphAssign %d\n" %get_ngVariables_to_assign(graph_def, all_ngVars))
    total_extra_added = total_extra_added + get_ngVariables_to_assign(graph_def, all_ngVars)

print("Total savings with current implementation in one iteration %d" %total_save_now)
print("Future savings in one iteration %d" %total_potential_save)
print("Total copies added due to NGraphAssign %d" %total_extra_added)







