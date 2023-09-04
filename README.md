# Dist-Saver

## Verifiable encryption using DIST-SAVER
This is a modified version of the [`SAVER`] protocol which was implemented [`here`] with the following implemented extensions:
- Distributed key generation. Allows multiple parties to combine their public keys and generate a common or group public key. using this key each party in the protocol can encrypt and verify in the same way as the original saver protocol However, no party alone will be able to decrypt that message, instead the homomorphic additive property of the underlying encryption (Elgamal encryption) can be used to combine multiple ciphertexts and generating an aggregate value. This final value can then be revealed to the destination party (the party that is allowed to access the aggregated result) using the next extension.
- Distributed re-encryption/key-switching. Switching the ciphertext from a plaintext encrypted with the common/group public key to a plaintext encrypted with the destination party's public key without reveal the underlying plaintext. 

### Testing
See the tests.rs file

[`SAVER`]: https://eprint.iacr.org/2019/1270
[`here`]: https://github.com/docknetwork/crypto

License: Apache-2.0
