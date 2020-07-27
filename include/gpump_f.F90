! Copyright (C) IBM Corporation 2018. All Rights Reserved
!
!    This program is licensed under the terms of the Eclipse Public License
!    v1.0 as published by the Eclipse Foundation and available at
!    http://www.eclipse.org/legal/epl-v10.html
!
!    
!    
! $COPYRIGHT$
module gpump_mod

interface gpump_f_init
subroutine gpump_f_init(comm) &
bind(C, name='gpump_f_init')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
end subroutine gpump_f_init
end interface gpump_f_init

interface gpump_f_register_region
subroutine gpump_f_register_region(mr_index, addr, size) &
bind(C, name='gpump_f_register_region')
use iso_c_binding
integer(kind=C_INT), intent(out) :: mr_index
integer(kind=C_LONG), intent(in), value :: addr
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_register_region
end interface gpump_f_register_region

interface gpump_f_replace_region
subroutine gpump_f_replace_region(mr_index, addr, size) &
bind(C, name='gpump_f_replace_region')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: addr
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_replace_region
end interface gpump_f_replace_region

interface gpump_f_deregister_region
subroutine gpump_f_deregister_region(mr_index) &
bind(C, name='gpump_f_deregister_region')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: mr_index
end subroutine gpump_f_deregister_region
end interface gpump_f_deregister_region

interface gpump_f_connect_propose
subroutine gpump_f_connect_propose(target) &
bind(C, name='gpump_f_connect_propose')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_connect_propose
end interface gpump_f_connect_propose

interface gpump_f_connect_accept
subroutine gpump_f_connect_accept(target) &
bind(C, name='gpump_f_connect_accept')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_connect_accept
end interface gpump_f_connect_accept

!interface gpump_f_connect
!subroutine gpump_f_connect(target) &
!bind(C, name='gpump_f_connect')
!use iso_c_binding
!integer(kind=C_INT), intent(in), value :: target
!end subroutine gpump_f_connect
!end interface gpump_f_connect

interface gpump_f_disconnect
subroutine gpump_f_disconnect(target) &
bind(C, name='gpump_f_disconnect')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_disconnect
end interface gpump_f_disconnect

interface gpump_f_create_window_propose
subroutine gpump_f_create_window_propose(target, local_address, remote_address, size) &
bind(C, name='gpump_f_create_window_propose')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_create_window_propose
end interface gpump_f_create_window_propose

interface gpump_f_replace_window_propose
subroutine gpump_f_replace_window_propose(target, local_address, remote_address, size) &
bind(C, name='gpump_f_replace_window_propose')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_replace_window_propose
end interface gpump_f_replace_window_propose

interface gpump_f_window_accept
subroutine gpump_f_window_accept(target) &
bind(C, name='gpump_f_window_accept')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_window_accept
end interface gpump_f_window_accept

!interface gpump_f_create_window
!subroutine gpump_f_create_window(target, local_address, remote_address, size) &
!bind(C, name='gpump_f_create_window')
!use iso_c_binding
!integer(kind=C_INT), intent(in), value :: target
!integer(kind=C_LONG), intent(in), value :: local_address
!integer(kind=C_LONG), intent(in), value :: remote_address
!integer(kind=C_LONG), intent(in), value :: size
!end subroutine gpump_f_create_window
!end interface gpump_f_create_window

interface gpump_f_destroy_window
subroutine gpump_f_destroy_window(target) &
bind(C, name='gpump_f_destroy_window')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_destroy_window
end interface gpump_f_destroy_window

interface gpump_f_create_window_propose_x
subroutine gpump_f_create_window_propose_x(target, wx, local_address, remote_address, size) &
bind(C, name='gpump_f_create_window_propose_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(out) :: wx
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_create_window_propose_x
end interface gpump_f_create_window_propose_x

interface gpump_f_replace_window_propose_x
subroutine gpump_f_replace_window_propose_x(target, wx, local_address, remote_address, size) &
bind(C, name='gpump_f_replace_window_propose_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: wx
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_replace_window_propose_x
end interface gpump_f_replace_window_propose_x

interface gpump_f_window_accept_x
subroutine gpump_f_window_accept_x(target, wx) &
bind(C, name='gpump_f_window_accept_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: wx
end subroutine gpump_f_window_accept_x
end interface gpump_f_window_accept_x

interface gpump_f_cork
subroutine gpump_f_cork() &
bind(C, name='gpump_f_cork')
use iso_c_binding
end subroutine gpump_f_cork
end interface gpump_f_cork

interface gpump_f_uncork
subroutine gpump_f_uncork(stream) &
bind(C, name='gpump_f_uncork')
use iso_c_binding
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_uncork
end interface gpump_f_uncork

interface gpump_f_stream_put
subroutine gpump_f_stream_put(target, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_put')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_put
end interface gpump_f_stream_put

interface gpump_f_iput
subroutine gpump_f_iput(target, offset, remote_offset, size) &
bind(C, name='gpump_f_iput')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iput
end interface gpump_f_iput

interface gpump_f_stream_wait_put_complete
subroutine gpump_f_stream_wait_put_complete(target, stream) &
bind(C, name='gpump_f_stream_wait_put_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_put_complete
end interface gpump_f_stream_wait_put_complete

interface gpump_f_cpu_ack_iput
subroutine gpump_f_cpu_ack_iput(target) &
bind(C, name='gpump_f_cpu_ack_iput')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_cpu_ack_iput
end interface gpump_f_cpu_ack_iput

interface gpump_f_is_put_complete
subroutine gpump_f_is_put_complete(target, is_complete) &
bind(C, name='gpump_f_is_put_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_put_complete
end interface gpump_f_is_put_complete

interface gpump_f_wait_put_complete
subroutine gpump_f_wait_put_complete(target) &
bind(C, name='gpump_f_wait_put_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_put_complete
end interface gpump_f_wait_put_complete

interface gpump_f_stream_get
subroutine gpump_f_stream_get(target, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_get')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_get
end interface gpump_f_stream_get

interface gpump_f_iget
subroutine gpump_f_iget(target, offset, remote_offset, size) &
bind(C, name='gpump_f_iget')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iget
end interface gpump_f_iget

interface gpump_f_stream_wait_get_complete
subroutine gpump_f_stream_wait_get_complete(target, stream) &
bind(C, name='gpump_f_stream_wait_get_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_get_complete
end interface gpump_f_stream_wait_get_complete

interface gpump_f_cpu_ack_iget
subroutine gpump_f_cpu_ack_iget(target) &
bind(C, name='gpump_f_cpu_ack_iget')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_cpu_ack_iget
end interface gpump_f_cpu_ack_iget

interface gpump_f_is_get_complete
subroutine gpump_f_is_get_complete(target, is_complete) &
bind(C, name='gpump_f_is_get_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_get_complete
end interface gpump_f_is_get_complete

interface gpump_f_wait_get_complete
subroutine gpump_f_wait_get_complete(target) &
bind(C, name='gpump_f_wait_get_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_get_complete
end interface gpump_f_wait_get_complete

interface gpump_f_stream_put_x
subroutine gpump_f_stream_put_x(target, index, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_put_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_put_x
end interface gpump_f_stream_put_x

interface gpump_f_iput_x
subroutine gpump_f_iput_x(target, index, offset, remote_offset, size) &
bind(C, name='gpump_f_iput_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iput_x
end interface gpump_f_iput_x

interface gpump_f_stream_wait_put_complete_x
subroutine gpump_f_stream_wait_put_complete_x(target, index, stream) &
bind(C, name='gpump_f_stream_wait_put_complete_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_put_complete_x
end interface gpump_f_stream_wait_put_complete_x

interface gpump_f_cpu_ack_iput_x
subroutine gpump_f_cpu_ack_iput_x(target, index) &
bind(C, name='gpump_f_cpu_ack_iput_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_cpu_ack_iput_x
end interface gpump_f_cpu_ack_iput_x

interface gpump_f_is_put_complete_x
subroutine gpump_f_is_put_complete_x(target, index, is_complete) &
bind(C, name='gpump_f_is_put_complete_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_put_complete_x
end interface gpump_f_is_put_complete_x

interface gpump_f_wait_put_complete_x
subroutine gpump_f_wait_put_complete_x(target, index) &
bind(C, name='gpump_f_wait_put_complete_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_wait_put_complete_x
end interface gpump_f_wait_put_complete_x

interface gpump_f_stream_get_x
subroutine gpump_f_stream_get_x(target, index, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_get_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_get_x
end interface gpump_f_stream_get_x

interface gpump_f_iget_x
subroutine gpump_f_iget_x(target, index, offset, remote_offset, size) &
bind(C, name='gpump_f_iget_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iget_x
end interface gpump_f_iget_x

interface gpump_f_stream_wait_get_complete_x
subroutine gpump_f_stream_wait_get_complete_x(target, index, stream) &
bind(C, name='gpump_f_stream_wait_get_complete_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_get_complete_x
end interface gpump_f_stream_wait_get_complete_x

interface gpump_f_cpu_ack_iget_x
subroutine gpump_f_cpu_ack_iget_x(target, index) &
bind(C, name='gpump_f_cpu_ack_iget_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_cpu_ack_iget_x
end interface gpump_f_cpu_ack_iget_x

interface gpump_f_is_get_complete_x
subroutine gpump_f_is_get_complete_x(target, index, is_complete) &
bind(C, name='gpump_f_is_get_complete_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_get_complete_x
end interface gpump_f_is_get_complete_x

interface gpump_f_wait_get_complete_x
subroutine gpump_f_wait_get_complete_x(target, index) &
bind(C, name='gpump_f_wait_get_complete_x')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_wait_get_complete_x
end interface gpump_f_wait_get_complete_x

interface gpump_f_stream_send
subroutine gpump_f_stream_send(target, stream, mr_index, offset, size) &
bind(C,name='gpump_f_stream_send')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_send
end interface gpump_f_stream_send

interface gpump_f_isend
subroutine gpump_f_isend(target, mr_index, offset, size) &
bind(C,name='gpump_f_isend')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_isend
end interface gpump_f_isend

interface gpump_f_stream_wait_send_complete
subroutine gpump_f_stream_wait_send_complete(target, stream) &
bind(C, name='gpump_f_stream_wait_send_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_send_complete
end interface gpump_f_stream_wait_send_complete

interface gpump_f_cpu_ack_isend
subroutine gpump_f_cpu_ack_isend(target) &
bind(C, name='gpump_f_cpu_ack_isend')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_cpu_ack_isend
end interface gpump_f_cpu_ack_isend

interface gpump_f_is_send_complete
subroutine gpump_f_is_send_complete(target, is_complete) &
bind(C, name='gpump_f_is_send_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_send_complete
end interface gpump_f_is_send_complete

interface gpump_f_wait_send_complete
subroutine gpump_f_wait_send_complete(target) &
bind(C, name='gpump_f_wait_send_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_send_complete
end interface gpump_f_wait_send_complete

interface gpump_f_receive
subroutine gpump_f_receive(target, mr_index, offset, size) &
bind(C,name='gpump_f_receive')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_receive
end interface gpump_f_receive

interface gpump_f_stream_wait_recv_complete
subroutine gpump_f_stream_wait_recv_complete(source, stream) &
bind(C, name='gpump_f_stream_wait_recv_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: source
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_recv_complete
end interface gpump_f_stream_wait_recv_complete

interface gpump_f_cpu_ack_recv
subroutine gpump_f_cpu_ack_recv(source) &
bind(C, name='gpump_f_cpu_ack_recv')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: source
end subroutine gpump_f_cpu_ack_recv
end interface gpump_f_cpu_ack_recv

interface gpump_f_is_receive_complete
subroutine gpump_f_is_receive_complete(target, is_complete) &
bind(C, name='gpump_f_is_receive_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_receive_complete
end interface gpump_f_is_receive_complete

interface gpump_f_wait_receive_complete
subroutine gpump_f_wait_receive_complete(target) &
bind(C, name='gpump_f_wait_receive_complete')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_receive_complete
end interface gpump_f_wait_receive_complete

interface gpump_f_term
subroutine gpump_f_term() &
bind(C, name='gpump_f_term')
end subroutine gpump_f_term
end interface gpump_f_term


interface gpump_f_init_r
subroutine gpump_f_init_r(comm) &
bind(C, name='gpump_f_init_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
end subroutine gpump_f_init_r
end interface gpump_f_init_r

interface gpump_f_register_region_r
subroutine gpump_f_register_region_r(comm, mr_index, addr, size) &
bind(C, name='gpump_f_register_region_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(out) :: mr_index
integer(kind=C_LONG), intent(in), value :: addr
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_register_region_r
end interface gpump_f_register_region_r

interface gpump_f_replace_region_r
subroutine gpump_f_replace_region_r(comm, mr_index, addr, size) &
bind(C, name='gpump_f_replace_region_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: addr
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_replace_region_r
end interface gpump_f_replace_region_r

interface gpump_f_deregister_region_r
subroutine gpump_f_deregister_region_r(comm, mr_index) &
bind(C, name='gpump_f_deregister_region_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: mr_index
end subroutine gpump_f_deregister_region_r
end interface gpump_f_deregister_region_r

interface gpump_f_connect_propose_r
subroutine gpump_f_connect_propose_r(comm, target) &
bind(C, name='gpump_f_connect_propose_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_connect_propose_r
end interface gpump_f_connect_propose_r

interface gpump_f_connect_accept_r
subroutine gpump_f_connect_accept_r(comm, target) &
bind(C, name='gpump_f_connect_accept_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_connect_accept_r
end interface gpump_f_connect_accept_r

!interface gpump_f_connect_r
!subroutine gpump_f_connect_r(comm, target) &
!bind(C, name='gpump_f_connect_r')
!use iso_c_binding
!integer(kind=C_INT), intent(in), value :: target
!end subroutine gpump_f_connect_r
!end interface gpump_f_connect_r

interface gpump_f_disconnect_r
subroutine gpump_f_disconnect_r(comm, target) &
bind(C, name='gpump_f_disconnect_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_disconnect_r
end interface gpump_f_disconnect_r

interface gpump_f_create_window_propose_r
subroutine gpump_f_create_window_propose_r(comm, target, local_address, remote_address, size) &
bind(C, name='gpump_f_create_window_propose_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_create_window_propose_r
end interface gpump_f_create_window_propose_r

interface gpump_f_replace_window_propose_r
subroutine gpump_f_replace_window_propose_r(comm, target, local_address, remote_address, size) &
bind(C, name='gpump_f_replace_window_propose_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_replace_window_propose_r
end interface gpump_f_replace_window_propose_r

interface gpump_f_window_accept_r
subroutine gpump_f_window_accept_r(comm, target) &
bind(C, name='gpump_f_window_accept_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_window_accept_r
end interface gpump_f_window_accept_r

!interface gpump_f_create_window_r
!subroutine gpump_f_create_window_r(comm, target, local_address, remote_address, size) &
!bind(C, name='gpump_f_create_window_r')
!use iso_c_binding
!integer(kind=C_INT), intent(in), value :: target
!integer(kind=C_LONG), intent(in), value :: local_address
!integer(kind=C_LONG), intent(in), value :: remote_address
!integer(kind=C_LONG), intent(in), value :: size
!end subroutine gpump_f_create_window_r
!end interface gpump_f_create_window_r

interface gpump_f_destroy_window_r
subroutine gpump_f_destroy_window_r(comm, target) &
bind(C, name='gpump_f_destroy_window_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_destroy_window_r
end interface gpump_f_destroy_window_r

interface gpump_f_create_window_propose_rx
subroutine gpump_f_create_window_propose_rx(comm, target, index, local_address, remote_address, size) &
bind(C, name='gpump_f_create_window_propose_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(out) :: index
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_create_window_propose_rx
end interface gpump_f_create_window_propose_rx

interface gpump_f_replace_window_propose_rx
subroutine gpump_f_replace_window_propose_rx(comm, target, index, local_address, remote_address, size) &
bind(C, name='gpump_f_replace_window_propose_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: local_address
integer(kind=C_LONG), intent(in), value :: remote_address
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_replace_window_propose_rx
end interface gpump_f_replace_window_propose_rx

interface gpump_f_window_accept_rx
subroutine gpump_f_window_accept_rx(comm, target, index) &
bind(C, name='gpump_f_window_accept_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_window_accept_rx
end interface gpump_f_window_accept_rx

!interface gpump_f_create_window_r
!subroutine gpump_f_create_window_r(comm, target, local_address, remote_address, size) &
!bind(C, name='gpump_f_create_window_r')
!use iso_c_binding
!integer(kind=C_INT), intent(in), value :: target
!integer(kind=C_LONG), intent(in), value :: local_address
!integer(kind=C_LONG), intent(in), value :: remote_address
!integer(kind=C_LONG), intent(in), value :: size
!end subroutine gpump_f_create_window_r
!end interface gpump_f_create_window_r

interface gpump_f_destroy_window_rx
subroutine gpump_f_destroy_window_rx(comm, target, index) &
bind(C, name='gpump_f_destroy_window_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_destroy_window_rx
end interface gpump_f_destroy_window_rx

interface gpump_f_cork_r
subroutine gpump_f_cork_r(comm) &
bind(C, name='gpump_f_cork_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
end subroutine gpump_f_cork_r
end interface gpump_f_cork_r

interface gpump_f_uncork_r
subroutine gpump_f_uncork_r(comm, stream) &
bind(C, name='gpump_f_uncork_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: stream
end subroutine gpump_f_uncork_r
end interface gpump_f_uncork_r

interface gpump_f_stream_put_r
subroutine gpump_f_stream_put_r(comm, target, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_put_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_put_r
end interface gpump_f_stream_put_r

interface gpump_f_iput_r
subroutine gpump_f_iput_r(comm, target, offset, remote_offset, size) &
bind(C, name='gpump_f_iput_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iput_r
end interface gpump_f_iput_r

interface gpump_f_stream_wait_put_complete_r
subroutine gpump_f_stream_wait_put_complete_r(comm, target, stream) &
bind(C, name='gpump_f_stream_wait_put_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_put_complete_r
end interface gpump_f_stream_wait_put_complete_r

interface gpump_f_cpu_ack_iput_r
subroutine gpump_f_cpu_ack_iput_r(comm, target) &
bind(C, name='gpump_f_cpu_ack_iput_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_cpu_ack_iput_r
end interface gpump_f_cpu_ack_iput_r

interface gpump_f_is_put_complete_r
subroutine gpump_f_is_put_complete_r(comm, target, is_complete) &
bind(C, name='gpump_f_is_put_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_put_complete_r
end interface gpump_f_is_put_complete_r

interface gpump_f_wait_put_complete_r
subroutine gpump_f_wait_put_complete_r(comm, target) &
bind(C, name='gpump_f_wait_put_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_put_complete_r
end interface gpump_f_wait_put_complete_r

interface gpump_f_stream_get_r
subroutine gpump_f_stream_get_r(comm, target, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_get_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_get_r
end interface gpump_f_stream_get_r

interface gpump_f_iget_r
subroutine gpump_f_iget_r(comm, target, offset, remote_offset, size) &
bind(C, name='gpump_f_iget_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iget_r
end interface gpump_f_iget_r

interface gpump_f_stream_wait_get_complete_r
subroutine gpump_f_stream_wait_get_complete_r(comm, target, stream) &
bind(C, name='gpump_f_stream_wait_get_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_get_complete_r
end interface gpump_f_stream_wait_get_complete_r

interface gpump_f_cpu_ack_iget_r
subroutine gpump_f_cpu_ack_iget_r(comm, target) &
bind(C, name='gpump_f_cpu_ack_iget_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_cpu_ack_iget_r
end interface gpump_f_cpu_ack_iget_r

interface gpump_f_is_get_complete_r
subroutine gpump_f_is_get_complete_r(comm, target, is_complete) &
bind(C, name='gpump_f_is_get_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_get_complete_r
end interface gpump_f_is_get_complete_r

interface gpump_f_wait_get_complete_r
subroutine gpump_f_wait_get_complete_r(comm, target) &
bind(C, name='gpump_f_wait_get_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_get_complete_r
end interface gpump_f_wait_get_complete_r

interface gpump_f_stream_put_rx
subroutine gpump_f_stream_put_rx(comm, target, index, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_put_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_put_rx
end interface gpump_f_stream_put_rx

interface gpump_f_iput_rx
subroutine gpump_f_iput_rx(comm, target, index, offset, remote_offset, size) &
bind(C, name='gpump_f_iput_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iput_rx
end interface gpump_f_iput_rx

interface gpump_f_stream_wait_put_complete_rx
subroutine gpump_f_stream_wait_put_complete_rx(comm, target, index, stream) &
bind(C, name='gpump_f_stream_wait_put_complete_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_put_complete_rx
end interface gpump_f_stream_wait_put_complete_rx

interface gpump_f_cpu_ack_iput_rx
subroutine gpump_f_cpu_ack_iput_rx(comm, target, index) &
bind(C, name='gpump_f_cpu_ack_iput_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_cpu_ack_iput_rx
end interface gpump_f_cpu_ack_iput_rx

interface gpump_f_is_put_complete_rx
subroutine gpump_f_is_put_complete_rx(comm, target, index, is_complete) &
bind(C, name='gpump_f_is_put_complete_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_put_complete_rx
end interface gpump_f_is_put_complete_rx

interface gpump_f_wait_put_complete_rx
subroutine gpump_f_wait_put_complete_rx(comm, target, index) &
bind(C, name='gpump_f_wait_put_complete_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_wait_put_complete_rx
end interface gpump_f_wait_put_complete_rx

interface gpump_f_stream_get_rx
subroutine gpump_f_stream_get_rx(comm, target, index, stream, offset, remote_offset, size) &
bind(C, name='gpump_f_stream_get_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_get_rx
end interface gpump_f_stream_get_rx

interface gpump_f_iget_rx
subroutine gpump_f_iget_rx(comm, target, index, offset, remote_offset, size) &
bind(C, name='gpump_f_iget_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: remote_offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_iget_rx
end interface gpump_f_iget_rx

interface gpump_f_stream_wait_get_complete_rx
subroutine gpump_f_stream_wait_get_complete_rx(comm, target, index, stream) &
bind(C, name='gpump_f_stream_wait_get_complete_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_get_complete_rx
end interface gpump_f_stream_wait_get_complete_rx

interface gpump_f_cpu_ack_iget_rx
subroutine gpump_f_cpu_ack_iget_rx(comm, target, index) &
bind(C, name='gpump_f_cpu_ack_iget_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_cpu_ack_iget_rx
end interface gpump_f_cpu_ack_iget_rx

interface gpump_f_is_get_complete_rx
subroutine gpump_f_is_get_complete_rx(comm, target, index, is_complete) &
bind(C, name='gpump_f_is_get_complete_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_get_complete_rx
end interface gpump_f_is_get_complete_rx

interface gpump_f_wait_get_complete_rx
subroutine gpump_f_wait_get_complete_rx(comm, target, index) &
bind(C, name='gpump_f_wait_get_complete_rx')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: index
end subroutine gpump_f_wait_get_complete_rx
end interface gpump_f_wait_get_complete_rx

interface gpump_f_stream_send_r
subroutine gpump_f_stream_send_r(comm, target, stream, mr_index, offset, size) &
bind(C,name='gpump_f_stream_send_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_stream_send_r
end interface gpump_f_stream_send_r

interface gpump_f_isend_r
subroutine gpump_f_isend_r(comm, target, mr_index, offset, size) &
bind(C,name='gpump_f_isend_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_isend_r
end interface gpump_f_isend_r

interface gpump_f_stream_wait_send_complete_r
subroutine gpump_f_stream_wait_send_complete_r(comm, target, stream) &
bind(C, name='gpump_f_stream_wait_send_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_send_complete_r
end interface gpump_f_stream_wait_send_complete_r

interface gpump_f_cpu_ack_isend_r
subroutine gpump_f_cpu_ack_isend_r(comm, target) &
bind(C, name='gpump_f_cpu_ack_isend_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_cpu_ack_isend_r
end interface gpump_f_cpu_ack_isend_r

interface gpump_f_is_send_complete_r
subroutine gpump_f_is_send_complete_r(comm, target, is_complete) &
bind(C, name='gpump_f_is_send_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_send_complete_r
end interface gpump_f_is_send_complete_r

interface gpump_f_wait_send_complete_r
subroutine gpump_f_wait_send_complete_r(comm, target) &
bind(C, name='gpump_f_wait_send_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_send_complete_r
end interface gpump_f_wait_send_complete_r

interface gpump_f_receive_r
subroutine gpump_f_receive_r(comm, target, mr_index, offset, size) &
bind(C,name='gpump_f_receive_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(in), value :: mr_index
integer(kind=C_LONG), intent(in), value :: offset
integer(kind=C_LONG), intent(in), value :: size
end subroutine gpump_f_receive_r
end interface gpump_f_receive_r

interface gpump_f_stream_wait_recv_complete_r
subroutine gpump_f_stream_wait_recv_complete_r(comm, source, stream) &
bind(C, name='gpump_f_stream_wait_recv_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: source
integer(kind=C_LONG), intent(in), value :: stream
end subroutine gpump_f_stream_wait_recv_complete_r
end interface gpump_f_stream_wait_recv_complete_r

interface gpump_f_cpu_ack_recv_r
subroutine gpump_f_cpu_ack_recv_r(comm, source) &
bind(C, name='gpump_f_cpu_ack_recv_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: source
end subroutine gpump_f_cpu_ack_recv_r
end interface gpump_f_cpu_ack_recv_r

interface gpump_f_is_receive_complete_r
subroutine gpump_f_is_receive_complete_r(comm, target, is_complete) &
bind(C, name='gpump_f_is_receive_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
integer(kind=C_INT), intent(inout) :: is_complete
end subroutine gpump_f_is_receive_complete_r
end interface gpump_f_is_receive_complete_r

interface gpump_f_wait_receive_complete_r
subroutine gpump_f_wait_receive_complete_r(comm, target) &
bind(C, name='gpump_f_wait_receive_complete_r')
use iso_c_binding
integer(kind=C_INT), intent(in), value :: comm
integer(kind=C_INT), intent(in), value :: target
end subroutine gpump_f_wait_receive_complete_r
end interface gpump_f_wait_receive_complete_r

interface gpump_f_term_r
subroutine gpump_f_term_r(comm) &
bind(C, name='gpump_f_term_r')
end subroutine gpump_f_term_r
end interface gpump_f_term_r

end module gpump_mod
