import math, random, numpy, sys
from numpy import median
we_need_neurons = ['S',3],['A',2],['F',2],['R',1] #setup
active_neurons = (
[0.1,0,0],
[0,0,0.9],
[0,0.8,0],
[0,1,1],
[1,0,0],
[1,0,1],
[1,1,0],
[1,1,1]
) # sensor neirons are activeted at start
correct_result = [0,1,0,0,1,1,0,0] # expected result
learnin_rate = 0.07 # how fast will network learning (jumping)
epoch = 800 # times of learning
total_errors = 0

neurons = [] # create list of neurons
layers_names = [] # we need to decide levels, witch could be different
for setup_layer in we_need_neurons:
    globals()[setup_layer[0]] = [] # now, we create A,S,R,etc lists
W = [] # list of sinapses

class Neuron:
    id = int
    fun = str
    activation_function_result = 0.0 # we need it to learning
    neuron_delta = 0.0
    neuron_error = 0

    def __init__(self): # add new neuron to neurons list
       self.id = len(neurons) + 1 # create neuron ID
       neurons.append(self)

    def activation_function(self):
        global W
        limit = 0.5
        if self.fun != we_need_neurons[0][0]: # not for sensor layer
            tW = 0.0
            for i in W:
                if i.connection_to == self:
                    tW += round(i.connection_from.activation_function_result) * i.weight
            sigmoid = 1/(1+math.exp(-tW))
            self.activation_function_result = sigmoid # save to neuron var
            if sigmoid >= limit: 
                return True
            else:
                return False
        else:
            if self.activation_function_result >= limit: return True 
            else: return False

    def find_neuron_error(self, stack_number):
        global W, correct_result
        if self.fun != 'R' and self.fun != 'S':
            for w in W:
                if w.connection_from == self:
                    self.neuron_error += w.connection_to.neuron_error * w.weight
                    # print(self.neuron_error)
        elif self.fun == 'R':
            self.neuron_error = correct_result[stack_number] - self.activation_function_result

    def derivative_of_sigmoid_fun(self):
        derivative_of_sigmoid_fun_result = self.activation_function_result * (1 - self.activation_function_result)
        self.neuron_delta = self.neuron_error * derivative_of_sigmoid_fun_result

class Sinapse:
    id = int
    connection_from = None
    connection_to = None
    weight = 0.0

    def __init__(self): # add new neuron to neurons list
       self.id = len(W) + 1 # create neuron ID
       W.append(self)
    
    def random_weight(self):
        n = random.random()
        self.weight = n


# now, let's create neurons, append them to lists and set them function
for n in we_need_neurons: # for every level
    for i in range (1, n[1] + 1): # becouse we start from one
        layer_name = n[0] 
        globals()[layer_name+'%s' % i] = Neuron() # create new neuron object
        globals()[layer_name].append(globals()[layer_name+'%s' % i]) # append to list
        globals()[layer_name][-1].fun = layer_name # set function of last element of neurons list
            
# lets activete every neuron we need to be activated at start by default
def activate_sensor_neurons(stack_number):
    global active_neurons, S
    stack_for_study = active_neurons[stack_number] 
    for i in range(len(stack_for_study)):
        for n in S:
            if n.id == i+1:
                n.activation_function_result = stack_for_study[i]

# now, we count how mutch sinapses does we need and create them
# we_need_sinapses = len(S) * len (A) + len (A) * len (R) # count how many sin we need
we_need_sinapses = 0
layer_n_count = [] # we need to decide levels, witch could be different
for setup_layer in we_need_neurons:
    layer_n_count.append(setup_layer[1])
for i in range(len(layer_n_count)-1): # level per level
    we_need_sinapses += layer_n_count[i] * layer_n_count[i+1]
for i in range (1, we_need_sinapses + 1): # becouse we start from one
    globals()['W%s' % i] = Sinapse() # create new sinapse object

def connect(): # connect every sinapses for every level
    global W, neurons
    used_connections = [] # to do not create the same connections
    layers_names = [] # we need to decide levels, witch could be different
    for setup_layer in we_need_neurons:
        layers_names.append(setup_layer[0])
    for i in range(len(layer_n_count)-1):
        neurons_from = globals()[layers_names[i]]
        neurons_to = globals()[layers_names[i+1]]
        for n_f in neurons_from:
            for n_t in neurons_to:
                for w in W:
                    if w.connection_from is None and (str(n_f.id) + " " + str(n_t.id)) not in used_connections:
                        w.connection_from = n_f
                        w.connection_to = n_t
                        used_connections.append(str(n_f.id) + " " + str(n_t.id))
                        w.random_weight() # set random weight

connect()

def spyke(): # activate or deactivate every neuron layer by layer
    global neurons, R1
    layer_names = [] # we need to decide levels, witch could be different
    for setup_layer in we_need_neurons:
        layer_names.append(setup_layer[0])
    for level in layer_names: # level per level
        # if level != 'S':
        for n in neurons:
            if n.fun == level:
                n.activation_function()
    return R1.activation_function()

def sort_by_id(obj): # not so easy to sort arreys of objects
    return obj.id

def learn(): # find eror, and correct weights using back propogation mehtod
    global neurons, W, R, R1, correct_result, learnin_rate, active_neurons
    results = []
    neurons = sorted(neurons, reverse = True) # was 1-2-3, now 3-2-1
    for stack_number in range(len(active_neurons)):
        activate_sensor_neurons(stack_number)
        spyke()
        
        neurons.sort(key = sort_by_id, reverse = True) # was 1-2-3, now 3-2-1
        for n in neurons:
            n.neuron_error = 0 # reset
            n.find_neuron_error(stack_number)
            n.derivative_of_sigmoid_fun()
        
        for w in W:
            weight = w.weight + w.connection_to.neuron_delta * w.connection_from.activation_function_result * learnin_rate
            w.weight = weight

        cr = correct_result[stack_number]
        error = cr - R1.activation_function_result
        result = numpy.mean(error**2)
        results.append(result)
    median_result = median(results)
    return median_result

def print_network():
    print('\rNetwork state:')
    for w in W:
        print('S ID{0:3}: {1:1}{2:3} - {3:1}{4:3}, W = {5:3}'.format(w.id, w.connection_from.fun, w.connection_from.id, w.connection_to.fun, w.connection_to.id, w.weight))
    for n in neurons:
        print('{0:1}{1:3} result: {2:3}'.format(n.fun, n.id, n.activation_function_result))
# activate_sensor_neurons(1)
# spyke()        
# print_network()


def learn_every_epoch(): # learning process throw all epoch
    global correct_result, epoch, learnin_rate
    learnin_rate_constant = learnin_rate
    devations = [] # network results
    for i in range(1,epoch+1):
        if i == 1:
            learnin_rate = learnin_rate_constant / 4
        if i == 5:
            learnin_rate = learnin_rate_constant * 5
        if i > epoch / 3 * 2:
            learnin_rate = learnin_rate_constant / 2
        result = learn() # errors rate after learning
        if result < 0.01:
            print('Learning has been stoped at epoch ' + str(i))
            break
        devations.append(result) # add
    print('\rDeviation history:')
    firts_result = devations[0]
    i = 0
    for r in devations:
        i += 1
        # if i % 100 == 0 or i == 1: # every 100 epoch
        print('   Epoch ' + str(i) + ': ' + str(round(r,4)))

learn_every_epoch()

def control_question():
    global active_neurons, correct_result, total_errors
    for stack_number in range(len(active_neurons)):
        activate_sensor_neurons(stack_number)
        spyke()
        print_network()
        active_n = active_neurons[stack_number]
        correct_res = correct_result[stack_number]
        prediction = R1.activation_function_result
        prediction_result = round(R1.activation_function_result)
        if prediction_result == correct_res: 
            result = 'CORRECT'
        else: 
            result = 'ERROR'
            total_errors += 1
        print('### CONTROL: for S = ' + str(active_n) + ' correct res = ' + str(correct_res) + '. Prediction is ' + str(round(prediction,4)) + '. Prediction result = ' + str(prediction_result) + ' | ' + result)
    if total_errors == 0:
        print('RESULT: NO ERRORS AFTER LEARNING')
    else:
        print('RESULT: HERE IS ' + str(total_errors) + ' ERRORS AFTER LEARNING')

control_question()
