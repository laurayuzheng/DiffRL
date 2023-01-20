'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''

import torch as th

IDM_DELTA = 4.0

class IDMLayer:
    '''
    Compute acceleration of a vehicle based on IDM model.
    Parameters are all PyTorch Tensors of shape [N, 1], where N is the number of vehicles, or batch.

     1. a_max: Maximum acceleration                                     
     2. a_pref: Preferred acceleration (or deceleration)
     3. v_curr: Current velocity of ego vehicle
     4. v_target: Target velocity of ego vehicle
     5. pos_delta: Position delta from leading vehicle
     6. vel_delta: Velocity delta from leading vehicle
     7. min_space: Minimum desired distance to leading vehicle
     8. time_pref: Desired time to move forward with current speed
     9. delta_time: Length of time step
    '''
    @staticmethod
    def apply(a_max: th.Tensor, 
                a_pref: th.Tensor, 
                v_curr: th.Tensor, 
                v_target: th.Tensor, 
                pos_delta: th.Tensor, 
                vel_delta: th.Tensor, 
                min_space: th.Tensor, 
                time_pref: th.Tensor,
                delta_time: th.Tensor):

        # Check input validity        
        IDMLayer.check_input(a_max, 
                            a_pref, 
                            v_curr, 
                            v_target, 
                            pos_delta, 
                            vel_delta, 
                            min_space, 
                            time_pref,
                            delta_time)
        
        optimal_spacing = IDMLayer.compute_optimal_spacing(
                            a_max, 
                            a_pref, 
                            v_curr, 
                            v_target, 
                            pos_delta, 
                            vel_delta, 
                            min_space, 
                            time_pref)

        # @BUGFIX: Optimal spacing cannot be negative.
        # If it is allowed to be, acceleration could become negative value even when the leading
        # vehicle is much faster than ego vehicle, so it can accelerate more.
        optimal_spacing = th.clip(optimal_spacing, min=0.0)
        
        acc = IDMLayer.compute_acceleration(
                                            a_max, 
                                            a_pref, 
                                            v_curr, 
                                            v_target, 
                                            pos_delta, 
                                            vel_delta, 
                                            min_space, 
                                            time_pref,
                                            optimal_spacing)

        # Clip negative velocities
        v_next = v_curr + acc * delta_time
        acc = th.where(v_next >= 0, acc, -v_curr / delta_time)
        
        return acc

    @staticmethod
    def check_input(a_max: th.Tensor, 
                    a_pref: th.Tensor, 
                    v_curr: th.Tensor, 
                    v_target: th.Tensor, 
                    pos_delta: th.Tensor, 
                    vel_delta: th.Tensor, 
                    min_space: th.Tensor, 
                    time_pref: th.Tensor,
                    delta_time: th.Tensor):
        num_batch = a_max.shape[0]

        assert a_max.ndim == 1, "Invalid Tensor shape"
        assert a_pref.ndim == 1, "Invalid Tensor shape"
        assert v_curr.ndim == 1, "Invalid Tensor shape"
        assert v_target.ndim == 1, "Invalid Tensor shape"
        assert pos_delta.ndim == 1, "Invalid Tensor shape"
        assert vel_delta.ndim == 1, "Invalid Tensor shape"
        assert min_space.ndim == 1, "Invalid Tensor shape"
        assert time_pref.ndim == 1, "Invalid Tensor shape"
        assert delta_time.ndim == 1, "Invalid Tensor shape"

        assert a_max.shape[0] == num_batch, "Invalid Tensor shape"
        assert a_pref.shape[0] == num_batch, "Invalid Tensor shape"
        assert v_curr.shape[0] == num_batch, "Invalid Tensor shape"
        assert v_target.shape[0] == num_batch, "Invalid Tensor shape"
        assert pos_delta.shape[0] == num_batch, "Invalid Tensor shape"
        assert vel_delta.shape[0] == num_batch, "Invalid Tensor shape"
        assert min_space.shape[0] == num_batch, "Invalid Tensor shape"
        assert delta_time.shape[0] == num_batch, "Invalid Tensor shape"

    @staticmethod
    def compute_optimal_spacing(a_max: th.Tensor, 
                                a_pref: th.Tensor, 
                                v_curr: th.Tensor, 
                                v_target: th.Tensor, 
                                pos_delta: th.Tensor, 
                                vel_delta: th.Tensor, 
                                min_space: th.Tensor, 
                                time_pref: th.Tensor):

        optimal_spacing = (min_space + v_curr * time_pref + \
            ((v_curr * vel_delta) / (2 * th.sqrt(a_max * a_pref))))

        return optimal_spacing

    @staticmethod
    def compute_acceleration(a_max, 
                            a_pref, 
                            v_curr, 
                            v_target, 
                            pos_delta, 
                            vel_delta, 
                            min_space, 
                            time_pref,
                            optimal_spacing):
        acc = a_max * (1.0 - th.pow(v_curr / v_target, IDM_DELTA) - \
            th.pow((optimal_spacing / pos_delta), 2.0))

        return acc